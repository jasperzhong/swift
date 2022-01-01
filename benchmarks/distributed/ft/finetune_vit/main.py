import argparse
import logging
import os
import random
import time
from torch.optim import lr_scheduler
import math
import numpy as np
from schedule import (get_num_microbatches, initialize_global_args,
                      is_pipeline_first_stage, is_pipeline_last_stage,
                      pipedream_flush_schedule)
from model import PipelineParallelViT
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
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
# lr scheduler
parser.add_argument('--lr', '--learning-rate', default=3e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-min', '--learning-rate-min', default=0, type=float,
                    help='min of learning rate (defalut 0)')
parser.add_argument('--warm-up-iters', default=10000, type=float,
                    help='warm up iterations')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--benchmark-iters', default=10000, type=int, metavar='N',
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
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((384, 384), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainset = datasets.CIFAR100(root=args.data,
                                train=True,
                                download=False)
    testset = datasets.CIFAR100(root=args.data,
                                train=False,
                                download=False,
                                transform=transform_test)

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.micro_batch_size,
                              num_workers=48,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.micro_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None
    return train_loader, test_loader


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


def train_iter(model, optimizer, data_iterator, loss_func, lr_scheduler=None):
    start = time.time()
    optimizer.zero_grad()
    loss = pipedream_flush_schedule(
        data_iterator, model, loss_func)
    torch.cuda.synchronize()
    # gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()
    iteration_time = time.time() - start
    return loss

def get_lr_scheduler(optimizer, total_iters, args):
    cosine_lr = lambda iter: 0.5 * ( math.cos((iter - args.warm_up_iters) /(total_iters - args.warm_up_iters) * math.pi) + 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr)
    return scheduler

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

    data_loader, test_loader = get_data_loader(args)
    model = PipelineParallelViT(balance=[1,3,3,3,3,2,2,1])
    model.cuda()

    micro_batch_num = args.global_batch_size // args.micro_batch_size
    total_iters = args.epochs * (len(data_loader) // micro_batch_num)
    print("total iterations: {}".format(total_iters))
    # lr rate???
    optimizer = optim.SGD(model.parameters(), lr=3e-3, momentum=0.9)
    lr_scheduler = get_lr_scheduler(optimizer, total_iters, args)
    loss_func = nn.CrossEntropyLoss().cuda()

    config = FaultToleranceConfig(
        num_iteration=total_iters, batch_size=args.global_batch_size, checkpoint_interval=10,
        replica=False, logging=args.logging, logging_compression=args.logging_compression,
        logging_chunk_freq=args.logging_chunk_freq,
        logging_dfs=args.logging_dfs, logging_bucket=args.logging_s3_bucket,
        logging_group_size=args.logging_group_size, logging_groups=None, print_freq=args.print_freq
    )
    fault_tolerance_train(config, train_iter, model, optimizer,
                          data_loader, loss_func, lr_scheduler,
                          reset_data_iterator_func=reset_data_iterator)


if __name__ == '__main__':
    main()
