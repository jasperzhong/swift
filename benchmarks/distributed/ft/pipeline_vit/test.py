import argparse
import logging
import os
import random
import time
from benchmarks.distributed.ft.pipeline_vit.model import PipelineParallelViT
from torch.optim import lr_scheduler
import math
import numpy as np
from schedule import (get_num_microbatches, initialize_global_args,
                      is_pipeline_first_stage, is_pipeline_last_stage,
                      pipedream_flush_schedule)
from torchvision import datasets, transforms

import torch
import torch.distributed.fault_tolerance
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fault_tolerance import FaultToleranceConfig, fault_tolerance_train, warmup_profile
from torch.utils.data.sampler import RandomSamplerFromIdx, SequentialSamplerFromIdx, SequentialSampler

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

    train_sampler = RandomSamplerFromIdx(train_dataset, 0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, 
        batch_size=args.micro_batch_size,
        num_workers=args.workers, pin_memory=True
    )
    return train_loader


def reset_data_iterator(config, data_loader, ts):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    train_dataset = data_loader.dataset
    idx = ts._value * config.num_microbatches * args.micro_batch_size
    train_sampler = RandomSamplerFromIdx(train_dataset, idx)
    # data_loader.sampler = train_sampler
    micro_batch_size = config.batch_size // config.num_microbatches
    data_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, 
        batch_size=args.micro_batch_size,
        num_workers=32, pin_memory=True
    )
    data_iterator = iter(data_loader)
    
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
    warm_up_with_cosine_lr = lambda iter: iter / args.warm_up_iters if iter <= args.warm_up_iters \
                            else 0.5 * ( math.cos((iter - args.warm_up_iters) /(total_iters - args.warm_up_iters) * math.pi) + 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    return scheduler

def main():

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data_loader = get_data_loader(args)
    data_iter = iter(data_loader)
    for i in range(20):
        _, label = next(data_iter)
    
    _, label = next(data_iter)
    print(label)

    torch.randn(1000)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_dataset = data_loader.dataset
    train_sampler = RandomSamplerFromIdx(train_dataset, 20 * args.micro_batch_size)
    
    data_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, 
        batch_size=args.micro_batch_size,
        num_workers=32, pin_memory=True
    )

    data_iter = iter(data_loader)
    _, label = next(data_iter)
    print(label)

    



if __name__ == '__main__':
    main()
