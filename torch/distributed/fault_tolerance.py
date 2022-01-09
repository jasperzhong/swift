import logging
import os
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from queue import Queue

import torch
import torch.distributed.distributed_c10d as distributed_c10d
import torch.nn
import torch.optim
from torch._C._distributed_c10d import SwiftInternalError

from .data_parallel import (DistributedOptimizer, broadcast_optimizer_state,
                            broadcast_parameters)
from .distributed_c10d import (_failure_handler, all_gather, barrier,
                               broadcast, destroy_process_group,
                               get_local_world_size, get_rank, get_world_size,
                               is_local_root_rank, new_group,
                               parallel_recovery_data_parallel_size)


logger = logging.getLogger(__name__)


def _need_recovery(groups, failure_workers):
    rank = get_rank()
    for group in groups:
        if rank in group:
            for failure_worker in failure_workers:
                if failure_worker in group:
                    return True
    return False


def _download_logging_files(logging_files):
    client = distributed_c10d._logging_dfs_client
    for i in range(len(logging_files)):
        while logging_files[i]:
            dfs_files = client.ls()
            for file in logging_files[i]:
                if file in dfs_files:
                    client.download(dfs_path=file, local_path=file)
                    logger.info(f"download {file} from dfs")
                    logging_files[i].remove(file)
            time.sleep(0.1)


def _whether_to_upload_logging_files(groups, failure_workers):
    pairs = groups_to_pairs(groups)
    rank = get_rank()
    for pair in pairs:
        if rank in pair:
            for failure_worker in failure_workers:
                if failure_worker in pair:
                    return True
    return False


def _get_checkpoint_path(config):
    rank = get_rank()
    return config.checkpoint_prefix + str(rank) + ".ckpt"


class FileInfo:
    def __init__(self, filename, file_object, valid_keys):
        self.filename = filename
        self.file_object = file_object
        self.valid_keys = valid_keys


def _set_recovery_mask(config, ts, consensus_value):
    logging_files = get_logging_files(config, ts, consensus_value)
    logger.info(f"logging_files: {logging_files}")
    download_thread = threading.Thread(target=_download_logging_files, args=(logging_files, ), daemon=True)
    download_thread.start()


class FaultToleranceConfig:
    def __init__(self, num_iteration, iters_per_epoch, batch_size, num_microbatches, checkpoint_interval, replica=False, data_parallel_size=None,
                 logging=False, parallel_recovery=False, logging_compression=None, logging_chunk_freq=None, logging_dfs=None, logging_bucket=None,
                 logging_group_size=None, logging_groups=None, print_freq=5, checkpoint_prefix="swift_"):
        self.num_iteration = num_iteration
        self.iters_per_epoch = iters_per_epoch
        self.batch_size = batch_size
        self.num_microbatches = num_microbatches
        self.checkpoint_interval = checkpoint_interval
        self.replica = replica
        self.data_parallel_size = data_parallel_size
        self.logging = logging
        self.parallel_recovery = parallel_recovery
        self.logging_compression = logging_compression
        self.logging_chunk_freq = logging_chunk_freq
        self.logging_dfs = logging_dfs
        self.logging_bucket = logging_bucket
        self.logging_group_size = logging_group_size
        self.logging_groups = logging_groups
        self.print_freq = print_freq
        self.checkpoint_prefix = checkpoint_prefix


def setup(config):
    if config.logging:
        config.groups = get_groups(config.logging_group_size, config.logging_groups)
        pairs = groups_to_pairs(config.groups)
        logger.info(pairs)
        if set_logging_mask(pairs):
            logger.info(f"enable logging on device {torch.cuda.current_device()}")
            distributed_c10d._logging_compression = config.logging_compression
            distributed_c10d._logging = True
            distributed_c10d._logging_stream = torch.cuda.Stream()
            distributed_c10d._logging_cpu_tensor_queue = Queue()
            distributed_c10d._logging_dfs_client = DFSClient.create(logging_dfs=config.logging_dfs,
                                                                    logging_bucket=config.logging_bucket)
            distributed_c10d._logging_thread = threading.Thread(
                target=distributed_c10d.flush_objects_to_dfs, args=(config, ))
            distributed_c10d._logging_thread.start()


def teardown(config):
    if distributed_c10d._logging:
        distributed_c10d._logging_cpu_tensor_queue.put(None)
        distributed_c10d._logging_thread.join()


# profile some time
init_time = 0
recovery_time = 0


def recovery(config, ts, model, optimizer, lr_scheduler=None):
    global init_time
    global recovery_time

    callback = None
    consensus_value, need_undo, failure_workers = ts.sync()
    logger.info(f"consensus_value is {consensus_value}")
    # init time end
    init_end = time.time()
    init_time = init_end - init_time
    logger.info(f"init time is: {init_time}")
    # only rank 0 get the init time
    if ts._value != 0 and get_rank() == 0:
        with open(f"main_init_{ts._value}.txt", "a") as f:
            f.write(f"{init_time}\n")
    logger.info(f"failure workers: {failure_workers}")
    recovery_time = time.time()

    if need_undo:
        logger.info(f"[rank {get_rank()}] undo update is needed"
                    f"(iteration = {consensus_value+1} while the consensus value is {consensus_value})!")

        if lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_scheduler.lr_lambdas[0], last_epoch=consensus_value - 1)
        optimizer.undo()

    old_optimizer = optimizer
    old_lr_scheduler = lr_scheduler
    if config.replica:
        barrier()
        N, d = get_world_size(), config.data_parallel_size
        pipeline_parallel_size = N // d
        pipeline_parallel_rank = get_rank() % pipeline_parallel_size

        ranks = list(zip(*[list(range(i * N // d, (i + 1) * N // d))
                           for i in range(d)]))[pipeline_parallel_rank]
        logger.info(f"ranks: {ranks}")
        comm = new_group(ranks, group_name="src_group_rank_%d_%d" % (pipeline_parallel_rank, consensus_value))
        if type(optimizer).__name__ == "DistributedOptimizer":
            optimizer.comm_group = comm
        else:
            optimizer = DistributedOptimizer(optimizer, model.named_parameters(),
                                             backward_passes_per_step=config.num_microbatches,
                                             comm_group=comm, average=True)

        group_src_rank = pipeline_parallel_rank
        if failure_workers:
            while group_src_rank in failure_workers:
                group_src_rank += pipeline_parallel_size
        logger.info(f"group src rank = {group_src_rank}")
        broadcast_parameters(model.state_dict(), group_src_rank, comm_group=comm)
        optimizer.clear()
        broadcast_optimizer_state(optimizer, group_src_rank, comm_group=comm)
        logger.info(f"Rank {group_src_rank} broadcast its parameters and optimizer states")
        # for failure worker
        ts._value = consensus_value
    elif config.logging:
        # upload needed logging files
        if _whether_to_upload_logging_files(config.groups, failure_workers):
            distributed_c10d._logging_cpu_tensor_queue.put("flush")

        need_recovery = _need_recovery(config.groups, failure_workers)
        if need_recovery:
            filename = _get_checkpoint_path(config)
            load_checkpoint(filename, ts, model, optimizer)
            _set_recovery_mask(config, ts, consensus_value)
            # recovery starts
            distributed_c10d._logging_in_recovery = True

        if failure_workers and config.parallel_recovery:
            distributed_c10d._logging_parallel_recovery = True
            # recovery starts
            distributed_c10d._logging_in_recovery = True

            enable_logging_on_disabled_worker = False
            if not distributed_c10d._logging:
                enable_logging_on_disabled_worker = True
                logger.info(f"enable logging on device {torch.cuda.current_device()}")
                distributed_c10d._logging_compression = config.logging_compression
                distributed_c10d._logging = True
                distributed_c10d._logging_dfs_client = DFSClient.create(logging_dfs=config.logging_dfs,
                                                                        logging_bucket=config.logging_bucket)

            # 1. living workers do checkpoint
            # logging_rng_state_cnt_bck = distributed_c10d._logging_rng_state_cnt
            if not need_recovery:
                filename = _get_checkpoint_path(config)
                # do not do gc here
                checkpoint(filename, ts, model, optimizer, garbage_collection=False)

                # close its own rng state file
                # if distributed_c10d._logging_rng_state_fd is not None:
                #     distributed_c10d._logging_rng_state_fd.close()
                #     distributed_c10d._logging_rng_state_fd = None
            else:
                pass
                # filename = "rng_state_%d.h5" % (get_rank())
                # f = h5py.File(filename, "r")
                # logging_rng_state_cnt_bck = int(sorted(f.keys(), key=lambda x: int(x))[-1])
                # f.close()
                # distributed_c10d._logging_dfs_client.upload(dfs_path=filename, local_path=filename)
                # logger.info(f"put {filename} on dfs")

            # 2. all workers build new comm group
            peer_failure_worker = get_peer_failure_worker(config, failure_workers)
            comm = build_communication_group(config, peer_failure_worker)
            group_rank = get_rank(group=comm)
            group_size = get_world_size(group=comm)
            distributed_c10d._logging_group_rank = group_rank
            distributed_c10d._logging_group_size = group_size
            distributed_c10d._logging_group_diff = get_rank() - peer_failure_worker
            logger.info(f"build new communication group ({group_rank} / {group_size})")

            # 3. living workers build model and optimizer
            optimizer_cls = optimizer.__class__
            optimizer_defaults = optimizer.defaults
            model, optimizer = build_model_and_optimizer(config, model,
                                                         optimizer, comm,
                                                         peer_failure_worker)

            # 4. broadcast failure worker's ts
            ts.broadcast(peer_failure_worker)

            if lr_scheduler:
                # It seems that when last_epoch != -1, the optimizer should has initialized the `initial_lr`
                # but in parallel recovery, the re-built optimizer does not have that attribute, so we need
                # to set the `initial_lr` by building it twice.
                # KeyError: "param 'initial_lr' is not specified in param_groups[0] when resuming an optimizer"
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=old_lr_scheduler.lr_lambdas[0], last_epoch=-1)

                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=old_lr_scheduler.lr_lambdas[0], last_epoch=ts - 1)
                logger.info(f"reset lr scheduler: lr={lr_scheduler.get_last_lr()}")

            # 5. hijack get_rank()
            get_rank_bck = torch.distributed.get_rank
            torch.distributed.get_rank = lambda group=None: peer_failure_worker
            logger.info(f"rank {get_rank_bck()} changes the rank to {torch.distributed.get_rank()}")

            # 6. download the same set of logging files as the failure worker
            if not need_recovery:
                # filename = "rng_state_%d.h5" % (peer_failure_worker)
                # while True:
                #     dfs_files = distributed_c10d._logging_dfs_client.ls()
                #     if filename in dfs_files:
                #         distributed_c10d._logging_dfs_client.download(dfs_path=filename, local_path=filename)
                #         break
                #     time.sleep(0.1)
                # logger.info(f"download {filename}")

                logging_files = get_logging_files_for_parallel_recovery(config, ts, consensus_value,
                                                                        peer_failure_worker)
                logger.info(f"logging_files: {logging_files}")
                if logging_files:
                    download_thread = threading.Thread(target=_download_logging_files, args=(logging_files, ),
                                                       daemon=True)
                    download_thread.start()

            def _cb(ts):
                nonlocal model
                nonlocal optimizer
                nonlocal old_optimizer
                nonlocal old_lr_scheduler
                nonlocal enable_logging_on_disabled_worker

                # recovery is over
                distributed_c10d._logging_in_recovery = False

                # close files
                for peer, item in distributed_c10d._logging_recovery_mask.items():
                    idx, file_info_list, consensus_value = item
                    logger.info(f"close file in cb. src={peer}")
                    f = file_info_list[idx].file_object
                    f.close()
                    logger.info("parallel recovery finishes")
                distributed_c10d._logging_recovery_mask.clear()
                distributed_c10d._logging_parallel_recovery = False
                distributed_c10d._logging_group_rank = None
                distributed_c10d._logging_group_size = None
                distributed_c10d._logging_group_diff = None

                # reload model and optimizer from checkpoint and reset logging mask after recovery
                torch.distributed.get_rank = get_rank_bck
                logger.info(f"rank {peer_failure_worker} changes the rank back to {torch.distributed.get_rank()}")
                model.assign_model_split(torch.distributed.get_rank())
                model.cuda()

                optimizer.remove_hooks()
                if not need_recovery:
                    optimizer = optimizer_cls(model.parameters(), **optimizer_defaults)
                    filename = _get_checkpoint_path(config)
                    load_checkpoint(filename, ts, model, optimizer)
                else:
                    # copy states from DistributedOptimizer
                    old_optimizer.load_state_dict(optimizer.state_dict())
                    optimizer = old_optimizer

                # # destroy communication group
                # destroy_process_group(comm)

                # disable logging
                if enable_logging_on_disabled_worker:
                    enable_logging_on_disabled_worker = False
                    distributed_c10d._logging_compression = None
                    distributed_c10d._logging = False
                    distributed_c10d._logging_dfs_client = None

                # # close its own rng state file
                # if distributed_c10d._logging_rng_state_fd is not None:
                #     distributed_c10d._logging_rng_state_fd.close()
                #     distributed_c10d._logging_rng_state_fd = None

                # distributed_c10d._logging_rng_state_cnt = logging_rng_state_cnt_bck + 1

                distributed_c10d._logging_mask.clear()
                # pairs = groups_to_pairs(config.groups)
                # set_logging_mask(pairs)
                return ts, model, optimizer, old_lr_scheduler

            callback = _cb
        else:
            def _cb(ts):
                nonlocal model
                nonlocal optimizer
                nonlocal old_lr_scheduler

                # recovery is over
                distributed_c10d._logging_in_recovery = False
                return ts, model, optimizer, old_lr_scheduler

            callback = _cb

    # global checkpointing
    else:
        filename = _get_checkpoint_path(config)
        load_checkpoint(filename, ts, model, optimizer)
        if lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_scheduler.lr_lambdas[0], last_epoch=ts - 1)
            logger.info(f"reset lr scheduler: lr={lr_scheduler.get_last_lr()}")

    return ts, model, optimizer, lr_scheduler, consensus_value, callback


def get_peer_failure_worker(config, failure_workers):
    global_rank = get_rank()
    # find which group the failure worker belongs to
    failure_group = None
    for group in config.groups:
        for failure_worker in failure_workers:
            if failure_worker in group:
                failure_group = group
                break
        if failure_group:
            break

    # find which failure worker belongs to my data parallel group
    data_parallel_groups = list(zip(*config.groups))
    peer_failure_worker = None
    for group in data_parallel_groups:
        if global_rank in group:
            for worker in group:
                if worker in failure_group:
                    peer_failure_worker = worker
                    break
        if peer_failure_worker:
            break

    return peer_failure_worker


def build_model_and_optimizer(config, model, optimizer, comm, peer_failure_worker):
    global_rank = get_rank()
    if global_rank != peer_failure_worker:
        model.assign_model_split(peer_failure_worker)
        model.cuda()
        optimizer_cls = optimizer.__class__
        optimizer_defaults = optimizer.defaults
        optimizer = optimizer_cls(model.parameters(), **optimizer_defaults)

    num_microbatches = config.num_microbatches // parallel_recovery_data_parallel_size()
    logger.info(f"Rank {global_rank}'s num_microbatches = {num_microbatches}'")
    distributed_optimizer = DistributedOptimizer(optimizer, model.named_parameters(),
                                                 backward_passes_per_step=num_microbatches,
                                                 comm_group=comm, average=False)

    # peer failure worker broadcast its parameters and optimizer states
    # to other group members
    broadcast_parameters(model.state_dict(), peer_failure_worker, comm_group=comm)
    broadcast_optimizer_state(distributed_optimizer, peer_failure_worker, comm_group=comm)
    logger.info(f"Rank {peer_failure_worker} broadcast its parameters and optimizer states")

    return model, distributed_optimizer


def build_communication_group(config, peer_failure_worker):
    data_parallel_groups = list(zip(*config.groups))
    rank = get_rank()
    for group in data_parallel_groups:
        if rank in group:
            return new_group(group, group_name="src_group_rank_%d" % peer_failure_worker)


# def checksum(ts, model, optimizer):
#     model_sum = 0
#     grad_sum = 0
#     for param in model.parameters():
#         model_sum += torch.sum(param)
#         if param.grad is not None:
#             grad_sum += torch.sum(param.grad)
#
#     optimizer_sum = 0
#     for group in optimizer.param_groups:
#         for p in group['params']:
#             if p.grad is not None:
#                 state = optimizer.state[p]
#                 if 'momentum_buffer' in state:
#                     optimizer_sum += torch.sum(state['momentum_buffer'])
#
#     with open("debug.log", "a") as f:
#         f.write(f"{ts} {model_sum} {optimizer_sum} {grad_sum}\n")

def compute_logging_size(num_micro_batches, balance, file="../profile.txt", num_machines=16):
    workers_per_machine = 8
    with open(file) as f:
        lines = f.readlines()
        start = workers_per_machine
        activation_size = []
        for start in balance:
            line = lines[start]
            line = line.strip('\n')
            nums = line.split(" ")
            nums = [int(num) for num in nums]
            sum = 1
            for n in nums:
                sum *= n
            sum *= 4  # fp32
            sum *= num_micro_batches
            activation_size.append(sum)

    print(f"activation size: {activation_size}")
    return activation_size


def merge_groups(workload, threshold, bandwidth, checkpoint_interval, num_micro_batches, num_machines=16, workers_per_machine=8):
    # TODO: check how to read the file
    recovery_time = []
    activation_size = []
    workers_per_machine = get_local_world_size()
    print(f"workers_per_machine {workers_per_machine}")
    num_machines = get_world_size() // get_local_world_size()
    for i in range(num_machines):
        with open(f"{workload}_compute_time_{i}.txt", "r") as f:
            recovery_time.append(float(f.read()))

    activation_size = compute_logging_size(num_micro_batches, file="./profile.txt", num_machines=num_machines)

    activation_sum = sum(activation_size)
    group_size = len(recovery_time)
    assert group_size == num_machines, "initial group size must be equal to num of machines"
    group = [[i] for i in range(group_size)]
    while threshold > activation_sum:
        min_r_m = float('inf')
        merge_id = -1
        min_r_merge = 0
        min_m_merge = 0
        for i in range(group_size - 1):
            r_merge = recovery_time[i] + recovery_time[i + 1] + activation_size[i] / bandwidth
            delta_m = activation_size[i] * checkpoint_interval
            delta_r = r_merge * 2 * workers_per_machine / num_machines \
                - recovery_time[i] * workers_per_machine / num_machines \
                - recovery_time[i + 1] * workers_per_machine / num_machines

            r_m = delta_r / delta_m
            if r_m < min_r_m:
                min_r_m = r_m
                merge_id = i
                min_r_merge = r_merge
                min_m_merge = delta_m

        # udpate recovery time
        recovery_time[merge_id] = min_r_merge
        activation_sum -= min_m_merge
        group[merge_id] = group[merge_id].extend(group[merge_id + 1])
        del group[merge_id + 1]
        del activation_size[merge_id]
        del recovery_time[merge_id + 1]
        group_size -= 1

    print(f"the merged group is {group}")
    return group


def warmup_profile(train_iter, model, optimizer, data_iterator, loss_func, lr_scheduler, warmup_iters):
    sum = 0
    model.train()
    for i in range(5):
        if i == 0:
            continue
        start = time.time()
        _, compute_time_sum = train_iter(model, optimizer, data_iterator, loss_func, lr_scheduler)
        print(f"compute_time_sum:{compute_time_sum}")
        print(f"all time {time.time() - start}")

    rank = get_rank()
    workers_per_machine = get_local_world_size()

    if is_local_root_rank():
        with open(f"profile/compute_time_{rank // workers_per_machine}.txt", "a") as f:
            f.write(f"{compute_time_sum} \n")


def fault_tolerance_train(config, train_iter, model, optimizer, data_loader, loss_func,
                          lr_scheduler, reset_data_iterator_func, fault_tolerance_val=None, test_loader=None):
    global init_time
    global recovery_time
    setup(config)

    if not os.path.exists("time.log"):
        base_time = time.time()
        with open("time.log", "w") as f:
            f.write(str(base_time))
    else:
        with open("time.log", "r") as f:
            base_time = float(f.read())

    ts = Timestamp(0)
    distributed_c10d._ts = ts
    filename = _get_checkpoint_path(config)
    checkpoint(filename, ts, model, optimizer)
    while True:
        ts, model, optimizer, lr_scheduler, consensus_value, cb = recovery(config, ts, model, optimizer, lr_scheduler)
        s = time.time()
        data_iterator = reset_data_iterator_func(config, data_loader, ts % config.iters_per_epoch)
        print("reset data iterator time is:{}".format(time.time() - s))
        iter_time_avg = 0
        throughput_avg = 0
        # checksum(ts, model, optimizer)
        try:
            while True:
                try:
                    logger.info(f"start from iteration {ts}")
                    model.train()
                    num = ts % config.iters_per_epoch
                    for _ in range(ts, config.num_iteration):
                        if num >= config.iters_per_epoch:
                            raise StopIteration

                        # failure worker back to the consensus_value and finish recovery
                        if ts._value != 0 and ts._value == consensus_value and get_rank() == 8:
                            recovery_time = time.time() - recovery_time
                            logger.info("recovery time is: {}".format(recovery_time))
                            with open(f"main_recovery_{ts._value}.txt", "a") as f:
                                f.write(f"{recovery_time}\n")

                        # for experiment:
                        if ts._value == 300 and get_rank() == 8 and not os.path.exists("./temp.flag"):
                            with open("temp.flag", "a") as f:
                                f.write("Already killed\n")
                            os.system("ps aux | grep -i torch | grep -v grep | awk {'print $2'} | xargs kill -15")

                        start = time.time()
                        loss, _ = train_iter(model, optimizer, data_iterator, loss_func, lr_scheduler)
                        iteration_time = time.time() - start
                        ts += 1
                        num += 1
                        # this is okay because ts has been increased by one
                        if ts % config.checkpoint_interval == 0:
                            checkpoint_start = time.time()
                            filename = _get_checkpoint_path(config)
                            checkpoint(filename, ts, model, optimizer)
                            logger.info("checkpoint time is {}".format(time.time() - checkpoint_start))

                        iteration_time = time.time() - start

                        iter_time_avg += iteration_time
                        throughput = config.batch_size / iteration_time * parallel_recovery_data_parallel_size()
                        throughput_avg += throughput

                        # since the failure is on a basis of machines,
                        # so only the last worker in the machine will print the loss
                        if ts % config.print_freq == 0 and is_local_root_rank():
                            if lr_scheduler:
                                info = "[Iteration {}] loss: {:.6f} current throughput: {:.2f} avg throughput: {:.2f} iteration time: {:.3f} average iteration time: {:.3f} lr: {}".format(
                                    ts, loss, throughput, throughput_avg / ts._value, iteration_time, iter_time_avg / ts._value, lr_scheduler.get_last_lr())
                                logger.info(info)

                            else:
                                info = "[Iteration {}] loss: {:.6f} current throughput: {:.2f} avg throughput: {:.2f} iteration time: {:.3f} average iteration time: {:.3f}".format(
                                    ts, loss, throughput, throughput_avg / ts._value, iteration_time, iter_time_avg / ts._value)
                                logger.info(info)

                        # TODO: logging throughput, parallel, on failure worker
                        if ts % config.print_freq == 0 and get_rank() in [0, 8]:
                            write = "{} {:.2f} {:.2f} {:.2f} \n".format(
                                    ts, time.time() - base_time, throughput, throughput_avg / ts._value)
                            with open(f"main_throughput_{get_rank()}.txt", "a") as f:
                                f.write(write)

                        if ts == consensus_value and cb:
                            ts, model, optimizer, lr_scheduler = cb(ts)
                            del data_iterator
                            data_iterator = reset_data_iterator_func(config, data_loader, ts % config.iters_per_epoch)
                            logger.info(f"parallel recovery restores from iteration {ts}")
                            cb = None

                        # checksum(ts, model, optimizer)
                    break
                except StopIteration as e:
                    if fault_tolerance_val:
                        logger.info("start validation at iteration: {}".format(ts))
                        fault_tolerance_val(config, model, test_loader, loss_func)
                        data_iterator = reset_data_iterator_func(config, data_loader, 0)
            if fault_tolerance_val:
                logger.info("Finish Training for {} iterations".format(ts))
                fault_tolerance_val(config, model, test_loader, loss_func)

            break
        except SwiftInternalError as e:
            # init time start
            init_time = time.time()
            logger.info("catch an error: " + str(e))
            _failure_handler()
            # force data iterators' workers to exit
            del data_iterator

    teardown(config)


class Timestamp:
    def __init__(self, value):
        if value < 0:
            raise ValueError("timestamp must not be less than zero!")
        self._value = value

    def __index__(self):
        return self._value

    def __mod__(self, other):
        return self._value % other

    def __add__(self, other):
        return self._value + other

    def __iadd__(self, other):
        self._value += other
        return self

    def __sub__(self, other):
        return self._value - other

    def __isub__(self, other):
        self._value -= other
        return self

    def __mul__(self, other):
        return self._value * other

    def __imul__(self, other):
        self._value *= other
        return self

    def __div__(self, other):
        return self._value / other

    def __idiv__(self, other):
        self._value /= other
        return self

    def __lt__(self, other):
        return self._value < other

    def __le__(self, other):
        return self._value <= other

    def __gt__(self, other):
        return self._value > other

    def __ge__(self, other):
        return self._value >= other

    def __eq__(self, other):
        return self._value == other

    def __ne__(self, other):
        return self._value != other

    def __str__(self):
        return "%d" % int(self._value)

    def __repr__(self):
        return "timestamp(%d)" % int(self._value)

    def sync(self):
        """
        Return need_undo (bool), failure_workers (list)

        """
        tensor = torch.LongTensor(1).cuda()
        tensor_list = [torch.LongTensor(1).cuda() for _ in range(get_world_size())]
        tensor[0] = self._value
        all_gather(tensor_list, tensor)

        values = []
        for t in tensor_list:
            values.append(int(t[0].item()))

        consensus_value = 0
        need_undo = False
        failure_workers = []
        # all start from 0
        if not any(values):
            return consensus_value, need_undo, failure_workers

        for i, v in enumerate(values):
            if v == 0:
                failure_workers.append(i)

        consensus_value = self._make_consensus(values)
        if self._value == 0:
            # failure workers
            return consensus_value, need_undo, failure_workers
        else:
            # living workers
            need_undo = consensus_value == self._value - 1
            self._value = consensus_value
            return consensus_value, need_undo, failure_workers

    def _make_consensus(self, values):
        max_v = max(values)
        if max_v - 1 in values:
            return max_v - 1
        return max_v

    def broadcast(self, src_rank):
        tensor = torch.LongTensor(1).cuda()
        tensor[0] = self._value
        broadcast(tensor, src_rank)
        self._value = int(tensor[0].item())


class DFSClient(ABC):
    @classmethod
    def create(cls, *args, **kwargs):
        dfs = kwargs['logging_dfs']
        if dfs == "hdfs":
            return HDFSClient()
        elif dfs == "s3":
            return S3Client(**kwargs)
        else:
            raise ValueError(f"unknown dfs client: {dfs}")

    @abstractmethod
    def upload(self, dfs_path, local_path):
        raise NotImplementedError

    @abstractmethod
    def download(self, dfs_path, local_path):
        raise NotImplementedError

    @abstractmethod
    def ls(self):
        raise NotImplementedError

    @abstractmethod
    def rm(self, dfs_path):
        raise NotImplementedError


class HDFSClient(DFSClient):
    def __init__(self, *args, **kwargs):
        # please add hdfs commannd in the PATH
        self.hdfs_bin = "hdfs"

    def upload(self, dfs_path, local_path):
        while True:
            try:
                result = subprocess.run([self.hdfs_bin, "dfs", "-put", local_path, "/"], check=True)
                break
            except Exception as e:
                logger.info(f"upload {dfs_path} failed. retry. error: {e}")
                time.sleep(0.1)

    def download(self, dfs_path, local_path):
        if local_path in os.listdir():
            # File exists
            return

        while True:
            try:
                result = subprocess.run([self.hdfs_bin, "dfs", "-get", "/" + dfs_path, "."], check=True)
                break
            except Exception as e:
                logger.info(f"download {dfs_path} failed. retry. error: {e}")
                time.sleep(0.1)

    def ls(self):
        result = subprocess.getoutput("hdfs dfs -ls /")
        return [item.split(' ')[-1].lstrip('/') for item in result.split('\n')[1:]]

    def rm(self, dfs_path):
        result = subprocess.run([self.hdfs_bin, "dfs", "-rm", "/" + dfs_path])


try:
    import boto3

    class S3Client(DFSClient):
        def __init__(self, *args, **kwargs):
            if "logging_bucket" in kwargs:
                bucket = kwargs["logging_bucket"]
            else:
                raise ValueError("bucket not found")

            self.s3 = boto3.client('s3')
            rsp = self.s3.list_buckets()
            all_buckets = [bucket['Name'] for bucket in rsp['Buckets']]
            assert bucket in all_buckets
            self.bucket = bucket

        def upload(self, dfs_path, local_path):
            self.s3.upload_file(local_path, self.bucket, dfs_path)

        def download(self, dfs_path, local_path):
            self.s3.download_file(self.bucket, dfs_path, local_path)

        def ls(self):
            rsp = self.s3.list_objects(Bucket=self.bucket)
            key = 'Contents'
            if key in rsp:
                files = rsp[key]
                return [file['Key'] for file in files]
            return []

        def rm(self, dfs_path):
            self.s3.delete_object(Bucket=self.bucket, Key=dfs_path)
except ImportError:
    class S3Client(DFSClient):
        pass


def get_groups(group_size=None, groups=None):
    workers = get_world_size()
    workers_in_groups = []
    if groups:
        for group in groups:
            for worker in group:
                workers_in_groups.append(worker)
        # workers_in_groups = [worker for worker in group for group in groups]
        workers_in_groups = sorted(workers_in_groups)
        all_workers = list(range(workers))
        for lhs, rhs in zip(workers_in_groups, all_workers):
            if lhs != rhs:
                raise ValueError("wrong input groups")
        return groups

    if group_size is None:
        group_size = get_local_world_size()

    group_num = workers // group_size
    remainder = workers % group_size
    groups = []
    cnt = 0
    for _ in range(group_num):
        groups.append(list(range(cnt, cnt + group_size)))
        cnt += group_size
    if remainder:
        groups.append(list(range(cnt, cnt + remainder)))
    return groups


def groups_to_pairs(groups):
    pairs = []
    for i in range(len(groups) - 1):
        pairs.append((groups[i][-1], groups[i + 1][0]))
    return pairs


def set_logging_mask(pairs):
    rank = get_rank()
    ok = False
    for pair in pairs:
        if rank in pair:
            peer = pair[1] if pair[0] == rank else pair[0]
            distributed_c10d._logging_mask[peer] = True
            ok = True

    return ok


def get_logging_files(config, ts, consensus_value):
    ok = False
    rank = get_rank()
    logging_files = []
    for i, chunk in enumerate(range(ts, consensus_value, config.logging_chunk_freq)):
        logging_files.append([])
        for peer, _ in sorted(distributed_c10d._logging_mask.items()):
            filename = f"logging_{peer}_{rank}_{chunk}.h5"
            logging_files[i].append(filename)

            if peer not in distributed_c10d._logging_recovery_mask:
                # [idx, FileInfo list, consensus_value]
                distributed_c10d._logging_recovery_mask[peer] = [0, [], consensus_value]

            distributed_c10d._logging_recovery_mask[peer][1].append(FileInfo(
                filename=filename,
                file_object=None,
                valid_keys=None
            ))
    return logging_files


def get_logging_files_for_parallel_recovery(config, ts, consensus_value, peer_failure_worker):
    pairs = groups_to_pairs(config.groups)
    # reset logging mask
    distributed_c10d._logging_mask.clear()
    for pair in pairs:
        if peer_failure_worker in pair:
            peer = pair[1] if pair[0] == peer_failure_worker else pair[0]
            distributed_c10d._logging_mask[peer] = True

    logging_files = []
    for i, chunk in enumerate(range(ts, consensus_value, config.logging_chunk_freq)):
        logging_files.append([])
        for peer, _ in sorted(distributed_c10d._logging_mask.items()):
            filename = f"logging_{peer}_{peer_failure_worker}_{chunk}.h5"
            logging_files[i].append(filename)

            if peer not in distributed_c10d._logging_recovery_mask:
                # [idx, FileInfo list, consensus_value]
                distributed_c10d._logging_recovery_mask[peer] = [0, [], consensus_value]

            distributed_c10d._logging_recovery_mask[peer][1].append(FileInfo(
                filename=filename,
                file_object=None,
                valid_keys=None
            ))
    return logging_files


def checkpoint(filename, ts, model, optimizer, garbage_collection=True):
    if os.path.exists(filename):
        ckpt = torch.load(filename)
        if ts <= ckpt['ts']:
            logger.info("checkpoint aborted because there is already a newer checkpoint")
            # do not load checkpoint here!!!
            # load_checkpoint(filename, ts, model, optimizer)
            return

    torch.save({
        'ts': ts._value,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, filename)
    logger.info(f"save checkpoint in iteration {ts}")

    # garbage collection, remove all logging files
    # note that there is no need to close those files
    # because they are already outdated
    if distributed_c10d._logging and garbage_collection:
        distributed_c10d._logging_gpu_tensor_queue.clear()
        distributed_c10d._logging_cpu_tensor_queue.put("gc")

        # if os.path.exists("rng_state_%d" % (get_rank())):
        #     os.remove("rng_state_%d.h5" % (get_rank()))
        #     logger.info("remove outdated rng_state file")


def load_checkpoint(filename, ts, model, optimizer):
    checkpoint = torch.load(filename)
    ts._value = checkpoint['ts']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info(f"load checkpoint from iteration {ts}")
