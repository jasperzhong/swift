import argparse
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed.fault_tolerance
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fault_tolerance import (FaultToleranceConfig,
                                               fault_tolerance_train)
from torch.utils.data import (DataLoader, RandomSampler,
                              SequentialSampler)
from torchvision import datasets, transforms

from model import PipelineParallelViT
from schedule import (get_num_microbatches, initialize_global_args,
                      is_pipeline_first_stage, is_pipeline_last_stage,
                      pipedream_flush_schedule)
from torch.utils.data.sampler import RandomSamplerFromIdx
from validation import fault_tolerance_val

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
parser.add_argument('--warm-up-iters', default=500, type=float,
                    help='warm up iterations')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--img-size', default=224, type=int,
                    help='input size')
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
parser.add_argument('--test-batch-size', type=int,
                    default=256, help='Test batch size.')
# logging
parser.add_argument('--logging', default=False, action="store_true",
                    help='whether to enable logging.')
parser.add_argument('--parallel-recovery', default=False, action="store_true",
                    help='whether to enable parallel recovery.')
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
    trainset = datasets.CIFAR100(root=args.data,
                                 train=True,
                                 download=False,
                                 transform=transforms.ToTensor())
    testset = datasets.CIFAR100(root=args.data,
                                train=False,
                                download=False,
                                transform=transforms.ToTensor())

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.micro_batch_size,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.test_batch_size,
                             num_workers=args.workers,
                             pin_memory=True,
                             drop_last=True)
    return train_loader, test_loader

# def reset_data_iterator(config, data_loader, ts):
#     if args.seed is not None:
#         random.seed(args.seed)
#         np.random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         torch.cuda.manual_seed(args.seed)
#     train_dataset = data_loader.dataset
#     idx = ts * config.num_microbatches * args.micro_batch_size
#     train_sampler = RandomSamplerFromIdx(train_dataset, idx)
#     data_loader = torch.utils.data.DataLoader(
#         train_dataset, sampler=train_sampler, 
#         batch_size=args.micro_batch_size,
#         num_workers=32, pin_memory=True
#     )
#     data_iterator = iter(data_loader)
    
#     return data_iterator

def reset_data_iterator(config, data_loader, ts):
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
    loss, compute_time = pipedream_flush_schedule(
        data_iterator, model, loss_func)
    torch.cuda.synchronize()
    if type(optimizer).__name__ == "DistributedOptimizer":
        optimizer.synchronize()
        # gradient clipping should be right after gradient synchronization
        # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        with optimizer.skip_synchronize():
            optimizer.step()
    else:
        # gradient clipping
        # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()
    iteration_time = time.time() - start
    return loss, compute_time


def get_lr_scheduler(optimizer, total_iters, args):

    def warm_up_with_cosine_lr(iter): return iter / args.warm_up_iters if iter <= args.warm_up_iters \
        else 0.5 * (math.cos((iter - args.warm_up_iters) / (total_iters - args.warm_up_iters) * math.pi) + 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
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
    # model = PipelineParallelViT(balance=[4, 6, 5, 3])
    balance = [1 for i in range(12)]
    balance[0] = 5
    balance[-1] = 3
    model = PipelineParallelViT(balance=balance)
    # model = PipelineParallelViT(balance=[1,3,3,3,3,2,2,1])
    model.cuda()

    total_iters = args.benchmark_iters
    print("total iterations: {}".format(total_iters))
    num_micro_batches = args.global_batch_size // args.micro_batch_size
    iters_per_epoch = len(data_loader) // num_micro_batches
    print("iters per epoch:{}".format(iters_per_epoch))

    optimizer = optim.SGD(model.parameters(), lr=3e-2, momentum=0.9)
    lr_scheduler = get_lr_scheduler(optimizer, total_iters, args)
    loss_func = nn.CrossEntropyLoss().cuda()

    config = FaultToleranceConfig(
        num_iteration=total_iters, iters_per_epoch=iters_per_epoch, batch_size=args.global_batch_size, num_microbatches=get_num_microbatches(),
        checkpoint_interval=200, replica=False, logging=args.logging, parallel_recovery=args.parallel_recovery,
        logging_compression=args.logging_compression, logging_chunk_freq=args.logging_chunk_freq,
        logging_dfs=args.logging_dfs, logging_bucket=args.logging_s3_bucket,
        logging_group_size=args.logging_group_size, logging_groups=None, print_freq=args.print_freq
    )
    start = time.time()
    fault_tolerance_train(config, train_iter, model, optimizer,
                          data_loader, loss_func, lr_scheduler,
                          reset_data_iterator_func=reset_data_iterator, fault_tolerance_val=fault_tolerance_val, test_loader=test_loader)

    end = time.time()
    print("Training time is {}".format(end - start))

    time.sleep(100)

if __name__ == '__main__':
    main()
