import argparse
import logging
import os
import random
import time

import numpy as np
from model import PipelineParallelResNet50
from schedule import (get_num_microbatches, initialize_global_args,
                      is_pipeline_first_stage, is_pipeline_last_stage,
                      pipedream_flush_schedule)
from torchvision import datasets, transforms

import torch
import torch.distributed.fault_tolerance
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fault_tolerance import FaultToleranceConfig, fault_tolerance_train

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    description='Pipeline Parallel ResNet50 Arguments')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--benchmark-iters', default=100, type=int, metavar='N',
                    help='number of total iterations to run for benchmark')
parser.add_argument('--master_ip', default=None, type=str,
                    help='master ip for c10d')
parser.add_argument('--master_port', default=None, type=int,
                    help='master port for c10d')
# Pipeline parallelism
parser.add_argument('--micro-batch-size', type=int, default=None,
                    help='Batch size per model instance (local batch size).')
parser.add_argument('--global-batch-size', type=int,
                    default=256, help='Training batch size.')
parser.add_argument('--logging', default=False, action="store_true",
                    help='whether to enable logging.')
parser.add_argument('--logging-chunk-freq', type=int,
                    default=10, help='chunk logging files every N iterations.')
parser.add_argument('--logging-compression', default=None, type=str,
                    help='compression methods for logging')
parser.add_argument('--logging-dfs', default='hdfs', type=str,
                    help='distributed filesystem for logging')
parser.add_argument('--logging-s3-bucket', default=None, type=str,
                    help='s3 bucket if using s3 as logging store')
parser.add_argument('--logging-group-size', default=None, type=int,
                    help='group size for logging')
args = parser.parse_args()
initialize_global_args(args)


def get_data_loader(args):
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.micro_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    return train_loader


def reset_data_iterator(data_loader, ts):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    data_iterator = iter(data_loader)
    for _ in range(ts):
        if is_pipeline_first_stage() or is_pipeline_last_stage():
            for _ in range(get_num_microbatches()):
                next(data_iterator)
    return data_iterator


def train_iter(model, optimizer, data_iterator, loss_func):
    start = time.time()
    optimizer.zero_grad()
    loss = pipedream_flush_schedule(
        data_iterator, model, loss_func)
    torch.cuda.synchronize()
    optimizer.step()
    iteration_time = time.time() - start
    return loss


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl'
    )

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    data_loader = get_data_loader(args)
    model = PipelineParallelResNet50(balance=[4, 2, 2, 3])
    model.cuda()

    def hook_fn_backward(m, grad_input, grad_output):
        with open("debug_backward.log", "a") as f:
            f.write(f"{m._get_name()} {torch.sum(grad_input) {torch.sum(grad_output)}}")

    for module in model.model_split.named_modules():
        module.register_backward_hook(hook_fn_backward)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_func = nn.CrossEntropyLoss().cuda()

    config = FaultToleranceConfig(
        num_iteration=args.benchmark_iters, batch_size=args.global_batch_size, checkpoint_interval=100,
        replica=False, logging=args.logging, logging_compression=args.logging_compression,
        logging_chunk_freq=args.logging_chunk_freq,
        logging_dfs=args.logging_dfs, logging_bucket=args.logging_s3_bucket,
        logging_group_size=args.logging_group_size, logging_groups=None, print_freq=args.print_freq
    )
    fault_tolerance_train(config, train_iter, model, optimizer,
                          data_loader, loss_func, reset_data_iterator_func=reset_data_iterator)


if __name__ == '__main__':
    main()
