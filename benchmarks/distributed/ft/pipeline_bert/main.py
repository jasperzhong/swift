import argparse
import logging
import os
import random
import time

import h5py
import numpy as np
import torch
import torch.distributed.fault_tolerance
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fault_tolerance import (FaultToleranceConfig,
                                               fault_tolerance_train)
from torch.utils.data import (DataLoader, Dataset, SequentialSampler)

from model import PipelineParallelBert
from schedule import (PolyWarmUpScheduler, get_num_microbatches,
                      initialize_global_args, is_pipeline_first_stage,
                      is_pipeline_last_stage, pipedream_flush_schedule)

logging.basicConfig(level=logging.INFO)

# Required parameters
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
# addition
parser.add_argument("--checkpoint_activations", default=0, type=int,
                    help="Whether to perform checkpoint activations.")
parser.add_argument("--warmup_proportion", default=0.01, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                    "E.g., 0.1 = 10%% of training.")
parser.add_argument("--max_predictions_per_seq", default=80, type=int,
                    help="The maximum total of masked tokens in input sequence")
parser.add_argument("--max_steps", default=1000, type=float,
                    help="Total number of training steps to perform.")

args = parser.parse_args()
initialize_global_args(args)


def handle_train_dir(args):
    train_dir = os.path.join(args.data, 'train')
    files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if
             os.path.isfile(os.path.join(train_dir, f)) and 'training' in f]
    files.sort()
    num_files = len(files)
    # random.Random(args.seed + epoch).shuffle(files)
    f_start_id = 0
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > num_files:
        remainder = torch.distributed.get_world_size() % num_files
        data_file = files[(f_start_id * torch.distributed.get_world_size() +
                           torch.distributed.get_rank() + remainder * f_start_id) % num_files]
    else:
        data_file = files[(f_start_id * torch.distributed.get_world_size() + torch.distributed.get_rank()) % num_files]

    return data_file


def create_pretraining_dataset(args):
    data_file = handle_train_dir(args)
    train_data = pretraining_dataset(input_file=data_file, max_pred_length=args.max_predictions_per_seq)
    # train_sampler = RandomSampler(train_data)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.micro_batch_size,
                                  num_workers=args.workers, pin_memory=True)
    return train_dataloader


class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]


class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss


def get_input_shape(data_loader: DataLoader):
    data_iter = iter(data_loader)
    input_ids, segment_ids, input_mask, _, _ = next(data_iter)
    return input_ids.shape, segment_ids.shape, input_mask.shape


def prepare_model_and_optimizer(args):

    model = PipelineParallelBert(
        rank=torch.distributed.get_rank(),
        balance=None
    )

    # base on the NVIDIA example: 5e-5
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # TODO: args
    # lr_scheduler = PolyWarmUpScheduler(optimizer,
    #                                    warmup=args.warmup_proportion,
    #                                    total_steps=args.max_steps)

    # base on the config file: Vocab size
    loss_func = BertPretrainingCriterion(30528)

    return model, optimizer, None, loss_func


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


def train_iter(model, optimizer, data_iterator, loss_func, lr_scheduler):
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

    data_loader = create_pretraining_dataset(args)

    model, optimizer, lr_scheduler, loss_func = prepare_model_and_optimizer(args)
    model.cuda()
    loss_func.cuda()
    # TODO: lr_scheduler

    total_iters = args.benchmark_iters
    print("total iterations: {}".format(total_iters))
    num_micro_batches = args.global_batch_size // args.micro_batch_size
    iters_per_epoch = len(data_loader) // num_micro_batches
    print("iters per epoch:{}".format(iters_per_epoch))

    config = FaultToleranceConfig(
        num_iteration=total_iters, iters_per_epoch=iters_per_epoch, batch_size=args.global_batch_size, num_microbatches=get_num_microbatches(),
        checkpoint_interval=10, replica=False, logging=args.logging, parallel_recovery=args.parallel_recovery,
        logging_compression=args.logging_compression, logging_chunk_freq=args.logging_chunk_freq,
        logging_dfs=args.logging_dfs, logging_bucket=args.logging_s3_bucket,
        logging_group_size=args.logging_group_size, logging_groups=None, print_freq=args.print_freq
    )
    fault_tolerance_train(config, train_iter, model, optimizer,
                          data_loader, loss_func, None, reset_data_iterator_func=reset_data_iterator)


if __name__ == '__main__':
    main()
