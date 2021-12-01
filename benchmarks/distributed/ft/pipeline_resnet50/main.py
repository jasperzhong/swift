import argparse
import os
import random
import sys
import time

import numpy as np
from model import PipelineParallelResNet50
from schedule import (initialize_global_args, is_pipeline_first_stage,
                      is_pipeline_last_stage, pipedream_flush_schedule)
from torchvision import datasets, transforms

import torch
import torch.distributed.fault_tolerance
import torch.nn as nn
import torch.optim as optim

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


def get_data_iterator(args):
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
    data_iterator = iter(train_loader)
    return data_iterator


@torch.distributed.fault_tolerance.run
def train(state, args, data_iterator, model, optimizer, loss_func):
    print("start from epoch={} iter={}".format(state.epoch, state.iteration))
    for epoch in range(state.epoch, args.epochs):
        state.epoch = epoch
        iteration = 0
        while True:
            if iteration < state.iteration:
                if is_pipeline_first_stage() or is_pipeline_last_stage():
                    next(data_iterator)
                iteration += 1
                continue

            try:
                start = time.time()
                optimizer.zero_grad()
                loss = pipedream_flush_schedule(
                    data_iterator, model, loss_func)
                optimizer.step()
                iteration_time = time.time() - start

                iteration += 1
                if is_pipeline_last_stage() and iteration % args.print_freq == 0:
                    print("[Epoch {}/Iteration {}] loss: {:.4f} Throughput: {:.2f}".format(
                        epoch, iteration, loss, args.global_batch_size / (iteration_time)
                    ))

                if iteration == args.benchmark_iters:
                    sys.exit()

                state.iteration = iteration
            except StopIteration:
                break


def main():
    args = parser.parse_args()
    initialize_global_args(args)

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

    data_iterator = get_data_iterator(args)
    model = PipelineParallelResNet50(balance=[4, 2, 2, 3])
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_func = nn.CrossEntropyLoss().cuda()

    state = torch.distributed.fault_tolerance.State(epoch=0, iteration=0, optimizer=optimizer)
    train(state, args, data_iterator, model, optimizer, loss_func)


if __name__ == '__main__':
    main()