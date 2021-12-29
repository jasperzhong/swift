import argparse
import logging
import os
import random
import time
from benchmarks.distributed.ft.pipeline_vit.model import PipelineParallelViT
from vit import ViT
import numpy as np
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
    description='Pipeline Parallel ViT Arguments')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=3e-3, type=float,
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
    data_iter = iter(data_loader)
    # model = PipelineParallelViT()
    model = ViT(image_size=224, 
            patch_size=32, 
            num_classes=1000, 
            dim=1024, # base 768 ; large 1024 ; huge 1280 
            depth=24, # base 12 ; large 24 ; huge 36 
            heads=16, # base 12 ; large 16 ; huge 16
            mlp_dim=4096,
            pool = 'cls', 
            channels = 3, 
            dim_head = 64, 
            dropout = 0.1, 
            emb_dropout = 0.1)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.3)
    # TODO: Cosine LR scheduler
    loss_func = nn.CrossEntropyLoss().cuda()

    for i in range(args.benchmark_iters):
        start = time.time()
        optimizer.zero_grad()
        total_loss =0
        for _ in range(args.global_batch_size // args.micro_batch_size):
            images, labels  = next(data_iter)
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            loss = loss_func(output, labels)
            loss.backward()
            total_loss += loss.item() / args.micro_batch_size
        optimizer.step()
        end = time.time()
        iteration_time = end - start
        if i % 5 == 0:
            print("[Iteration {}] loss: {:.6f} throughput: {:.2f}\n".format(
                i, loss, args.global_batch_size / iteration_time))


if __name__ == '__main__':
    main()