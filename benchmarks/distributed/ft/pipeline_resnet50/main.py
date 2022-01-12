import argparse
from benchmarks.distributed.ft.pipeline_resnet50.validation import fault_tolerance_val
import logging
import os
import random
import time

import numpy as np
from model import PipelineParallelResNet50
from schedule import (get_num_microbatches, get_pipeline_model_parallel_rank,
                      initialize_global_args, is_pipeline_first_stage,
                      is_pipeline_last_stage, pipedream_flush_schedule,
                      get_data_parallel_rank)
from torchvision import datasets, transforms

import torch
import torch.distributed.fault_tolerance
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fault_tolerance import (FaultToleranceConfig,
                                               fault_tolerance_train)
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSamplerFromIdx

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

# Data parallelism
parser.add_argument('--data-parallel-size', type=int, default=None,
                    help='Data-parallel size')

# Pipeline parallelism
parser.add_argument('--micro-batch-size', type=int, default=None,
                    help='Batch size per model instance (local batch size).')
parser.add_argument('--test-batch-size', type=int, default=2048,
                    help='Batch size per model instance (local batch size).')
parser.add_argument('--global-batch-size', type=int,
                    default=256, help='Training batch size.')

# replica
parser.add_argument('--replica', default=False, action="store_true",
                    help='whether to enable replication recovery.')

args = parser.parse_args()
initialize_global_args(args)


def get_data_loader(args):
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
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
    
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=args.data_parallel_size,
                                                  rank=get_data_parallel_rank(), shuffle=True, seed=args.seed, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler, batch_size=args.micro_batch_size,
        num_workers=args.workers, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    return train_loader, test_loader

def get_lr_scheduler(optimizer, total_iters, args):

    def adjust_learning_rate(iter):
        iters_per_epoch = total_iters // 90
        if iter <= iters_per_epoch * 30:
            return 1
        if iter <= iters_per_epoch * 60:
            return 0.1
        elif iter <= iters_per_epoch * 90:
            return 0.01

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=adjust_learning_rate)
    return scheduler

# def reset_data_iterator(config, data_loader, ts):
#     if args.seed is not None:
#         random.seed(args.seed)
#         np.random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         torch.cuda.manual_seed(args.seed)
#     data_iterator = iter(data_loader)
#     for _ in range(ts):
#         if is_pipeline_first_stage() or is_pipeline_last_stage():
#             for _ in range(get_num_microbatches()):
#                 next(data_iterator)
#     return data_iterator

def reset_data_iterator(config, data_loader, ts):
    if args.seed is not None:
        print(f"seed is {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    train_dataset = data_loader.dataset
    # 8092 / 16 = 512
    idx = ts * 512

    train_sampler = torch.utils.data.DistributedSamplerFromIdx(train_dataset, num_replicas=args.data_parallel_size,
                                            rank=get_data_parallel_rank(), shuffle=True, seed=args.seed, drop_last=True, idx=idx)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, 
        batch_size=args.micro_batch_size,
        num_workers=32, pin_memory=True
    )
    data_iterator = iter(data_loader)
    
    return data_iterator


def train_iter(model, optimizer, data_iterator, loss_func, lr_scheduler):
    start = time.time()
    optimizer.zero_grad()
    loss, compute_time = pipedream_flush_schedule(
        data_iterator, model, loss_func)
    torch.cuda.synchronize()
    optimizer.step()
    iteration_time = time.time() - start
    return loss, compute_time


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    num_machines = args.world_size // args.local_world_size
    if args.data_parallel_size > num_machines:
        raise ValueError(
            f"data-parallel size ({args.data_parallel_size}) should not be large than the number of machines ({num_machines})")
    else:
        args.micro_batch_size //= args.data_parallel_size

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
    print(f"pipeline rank = {get_pipeline_model_parallel_rank()}")

    # balance = [4, 2, 2, 3]
    balance = [5, 1, 1, 4]
    # balance = [4, 1, 1, 1, 1, 3]
    model = PipelineParallelResNet50(rank=get_pipeline_model_parallel_rank(), balance=balance)
    model.cuda()

    num_micro_batches = args.global_batch_size // args.micro_batch_size // args.data_parallel_size
    iters_per_epoch = len(data_loader) // num_micro_batches
    print("iters per epoch:{}".format(iters_per_epoch))
    total_iters = 90 * iters_per_epoch
    print("total iterations: {}".format(total_iters))

    optimizer = optim.SGD(model.parameters(), lr=6.4, momentum=0.9, weight_decay=1e-4)
    if args.data_parallel_size == 1 and args.replica:
        logging.warn(f"Replicas are not available because data-parallel size is {args.data_parallel_size}")
        args.replica = False

    lr_scheduler = get_lr_scheduler(optimizer, total_iters, args)

    loss_func = nn.CrossEntropyLoss().cuda()

    config = FaultToleranceConfig(
        num_iteration=total_iters, iters_per_epoch=10, batch_size=args.global_batch_size, num_microbatches=get_num_microbatches(),
        checkpoint_interval=200, replica=args.replica, data_parallel_size=args.data_parallel_size, print_freq=args.print_freq
    )

    fault_tolerance_train(config, train_iter, model, optimizer,
                          data_loader, loss_func, lr_scheduler, reset_data_iterator_func=reset_data_iterator, 
                          fault_tolerance_val=fault_tolerance_val, test_loader=test_loader)


if __name__ == '__main__':
    main()
