import getpass
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from queue import Queue
from benchmarks.distributed.ft.pipeline_resnet50.schedule import is_pipeline_last_stage

from hdfs import InsecureClient

import torch
from torch.distributed import distributed_c10d
import torch.nn
import torch.optim
from torch._C._distributed_c10d import SwiftInternalError

from .data_parallel import (_DistributedOptimizer, broadcast_optimizer_state,
                            broadcast_parameters)
from .distributed_c10d import (_failure_handler, all_gather,
                               get_local_world_size, get_rank, get_world_size)

try:
    import boto3
except ImportError:
    pass

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


class FileInfo:
    def __init__(self, filename, file_object, valid_keys):
        self.filename = filename
        self.file_object = file_object
        self.valid_keys = valid_keys


def _set_recovery_mask(config, ts, consensus_value):
    logging_files = get_logging_files(config, ts, consensus_value)
    logger.info(logging_files)
    download_thread = threading.Thread(target=_download_logging_files, args=(logging_files, ), daemon=True)
    download_thread.start()


class FaultToleranceConfig:
    def __init__(self, num_iteration, batch_size, checkpoint_interval, replica=False, logging=False,
                 logging_compression=None, logging_chunk_freq=None, logging_dfs=None, logging_bucket=None,
                 logging_group_size=None, logging_groups=None, print_freq=5, checkpoint_path="swift.ckpt"):
        self.num_iteration = num_iteration
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.replica = replica
        self.logging = logging
        self.logging_compression = logging_compression
        self.logging_chunk_freq = logging_chunk_freq
        self.logging_dfs = logging_dfs
        self.logging_bucket = logging_bucket
        self.logging_group_size = logging_group_size
        self.logging_groups = logging_groups
        self.print_freq = print_freq
        self.checkpoint_path = checkpoint_path


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


def recovery(config, ts, model, optimizer, lr_scheduler=None):
    consensus_value, need_undo, failure_workers = ts.sync()
    logger.info(f"failure workers: {failure_workers}")

    if need_undo:
        logger.info(f"[Rank {get_rank()}] undo update is needed"
                    f"(iteration = {consensus_value+1} while the consensus value is {consensus_value})!")
        # lr_scheduler undo()
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler.lr_lambdas[0], last_epoch=consensus_value-2)
        # lr_scheduler.step()
        # undo lr first
        lr_scheduler.undo(consensus_value)
        optimizer.undo()

    if config.replica:
        broadcast_parameters(model.named_parameters(), 0)
        optimizer.clear()
        broadcast_optimizer_state(optimizer.state_dict(), 0)
    elif config.logging:
        if _need_recovery(config.groups, failure_workers):
            load_checkpoint(config, ts, model, optimizer, lr_scheduler)
            _set_recovery_mask(config, ts, consensus_value)
        else:
            # TODO: parallel recovery
            pass


def checksum(ts, model, optimizer):
    model_sum = 0
    for param in model.parameters():
        model_sum += torch.sum(param)

    optimizer_sum = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                state = optimizer.state[p]
                if 'momentum_buffer' in state:
                    optimizer_sum += torch.sum(state['momentum_buffer'])

    with open("debug.log", "a") as f:
        f.write(f"{ts} {model_sum} {optimizer_sum}\n")

def _get_checkpoint_path():
    rank = get_rank()
    return "swift" + str(rank) + ".ckpt"

def fault_tolerance_train(config, train_iter, model, optimizer, data_loader, loss_func,
                          lr_scheduler, reset_data_iterator_func):
    setup(config)

    ts = Timestamp(0)
    distributed_c10d._ts = ts
    checkpoint(_get_checkpoint_path(), ts, model, optimizer, lr_scheduler)
    while True:
        recovery(config, ts, model, optimizer)
        data_iterator = reset_data_iterator_func(data_loader, ts)
        checksum(ts, model, optimizer)
        try:
            iter_time_avg = 0
            logger.info(f"start from iteration {ts}")
            for _ in range(ts, config.num_iteration):
                start = time.time()
                loss = train_iter(model, optimizer, data_iterator, loss_func, lr_scheduler)
                iteration_time = time.time() - start
                iter_time_avg += iteration_time
                ts += 1
                checksum(ts, model, optimizer)

                if ts % config.print_freq == 0 and is_pipeline_last_stage():
                    logger.info("[Iteration {}] loss: {:.6f} throughput: {:.2f} average iteration time: {}".format(
                        ts, loss, config.batch_size / iteration_time, iter_time_avg / ts._value))

            break
        except SwiftInternalError as e:
            logger.info("catch an error: " + str(e))
            _failure_handler()

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
        host = os.environ["MASTER_ADDR"]
        user = getpass.getuser()
        self.client = InsecureClient(f"http://{host}:50070", user=user)

    def upload(self, dfs_path, local_path):
        self.client.upload("/" + dfs_path, local_path, overwrite=True)

    def download(self, dfs_path, local_path):
        self.client.download("/" + dfs_path, local_path)

    def ls(self):
        return self.client.list("/")

    def rm(self, dfs_path):
        self.client.delete("/" + dfs_path)


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


def get_groups(group_size=None, groups=None):
    workers = get_world_size()
    if groups:
        workers_in_groups = [worker for worker in group for group in groups]
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

def checkpoint(filename, ts, model, optimizer, lr_scheduler):
    if os.path.exists(filename):
        ckpt = torch.load(filename)
        if ts <= ckpt['ts']:
            logger.info("checkpoint aborted because there is already a newer checkpoint")
            load_checkpoint(filename, ts, model, optimizer, lr_scheduler)
            return

    torch.save({
        'ts': ts._value,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }, filename)
    logger.info(f"save checkpoint in iteration {ts}")


def load_checkpoint(filename, ts, model, optimizer, lr_scheduler):
    checkpoint = torch.load(filename)
    ts._value = checkpoint['ts']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    logger.info(f"load checkpoint from iteration {ts}")
