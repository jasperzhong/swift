import argparse
import logging
import os
import random
import time

import math
from torch.optim import lr_scheduler
import h5py
import numpy as np
import torch
import torch.distributed.fault_tolerance
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fault_tolerance import (FaultToleranceConfig,
                                               fault_tolerance_train)
from torch.utils.data import (DataLoader, Dataset, SequentialSampler)
from Squad import read_squad_examples, convert_examples_to_features
from tokenization import get_tokenizer
from model import PipelineParallelBert
from schedule import (get_num_microbatches,
                      initialize_global_args, is_pipeline_first_stage,
                      is_pipeline_last_stage, pipedream_flush_schedule)
from torch.utils.data import (DataLoader, RandomSampler, 
                              SequentialSampler,TensorDataset)
from validation import fault_tolerance_val

logging.basicConfig(level=logging.INFO)

# Required parameters
parser = argparse.ArgumentParser(
    description='Pipeline Parallel ResNet50 Arguments')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--predict-file', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
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
parser.add_argument('--test-batch-size', type=int,
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
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                    "E.g., 0.1 = 10%% of training.")
parser.add_argument("--max_predictions_per_seq", default=80, type=int,
                    help="The maximum total of masked tokens in input sequence")
parser.add_argument("--max_steps", default=1000, type=float,
                    help="Total number of training steps to perform.")
parser.add_argument('--output-dir', default='./',metavar='DIR', help='path to dataset')
parser.add_argument("--n_best_size", default=20, type=int,
                    help="The total number of n-best predictions to generate in the nbest_predictions.json "
                        "output file.")
parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument("--eval_script",
                        help="Script to evaluate squad predictions",
                        default="evaluate-v1.1.py",
                        type=str)
parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

args = parser.parse_args()
initialize_global_args(args)

def create_train_dataloader(args, tokenizer):
    train_examples = read_squad_examples(
            input_file=args.data, is_training=True, version_2_with_negative=False)
    
    train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)
    
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.micro_batch_size)
    
    return train_dataloader
    
def get_input_shape(data_loader: DataLoader):
    data_iter = iter(data_loader)
    input_ids, segment_ids, input_mask, _, _ = next(data_iter)
    return input_ids.shape, segment_ids.shape, input_mask.shape


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
    # TODO: maybe use another one
    # gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()
    iteration_time = time.time() - start
    return loss

def get_lr_scheduler(optimizer, total_iters, args):

    def warm_up_with_linear_lr(iter): return iter / args.warm_up_iters if iter <= args.warm_up_iters \
        else (iter - total_iters) / (args.warm_up_iters - total_iters)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_linear_lr)
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
    
    start = time.time()
    tokenizer = get_tokenizer()
    print("get token time : {}".format(time.time() - start))
    start = time.time()
    data_loader = create_train_dataloader(args, tokenizer)
    print("create dataloader time : {}".format(time.time() - start))
    input_ids, segment_ids, input_mask = get_input_shape(data_loader)
    print(input_ids, segment_ids, input_mask)
    
    start = time.time()
    model = PipelineParallelBert(
        rank=torch.distributed.get_rank(),
        balance=[1, 2, 2, 2, 2, 2, 2, 1]
    )    
    print("create model time : {}".format(time.time() - start))

    num_micro_batches = get_num_microbatches()
    iters_per_epoch = len(data_loader) // num_micro_batches
    num_iterations = args.epochs * iters_per_epoch
    print("total_iterations: {}".format(num_iterations))
     
    ## TODO: BERT Optimizer
    optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01)
    
    model.cuda()
    # TODO: choose which scheduler to use
    args.warm_up_iters = args.warmup_proportion * num_iterations
    lr_scheduler = get_lr_scheduler(optimizer, num_iterations, args)

    config = FaultToleranceConfig(
        num_iteration=num_iterations, iters_per_epoch=iters_per_epoch,batch_size=args.global_batch_size, num_microbatches=get_num_microbatches(),
        checkpoint_interval=100, replica=False, logging=args.logging, parallel_recovery=args.parallel_recovery,
        logging_compression=args.logging_compression, logging_chunk_freq=args.logging_chunk_freq,
        logging_dfs=args.logging_dfs, logging_bucket=args.logging_s3_bucket,
        logging_group_size=args.logging_group_size, logging_groups=None, print_freq=args.print_freq
    )

    start = time.time()
    # Doesn't need loss_func here.
    fault_tolerance_train(config, train_iter, model, optimizer,
                          data_loader, loss_func=None, lr_scheduler=lr_scheduler,
                          reset_data_iterator_func=reset_data_iterator, fault_tolerance_val=fault_tolerance_val, test_loader=None)

    end = time.time()
    print("Training time is {}".format(end - start))


if __name__ == '__main__':
    main()
