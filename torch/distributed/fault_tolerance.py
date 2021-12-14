import functools
import getpass
import logging
import os
import threading
from queue import Queue
from abc import ABC, abstractmethod

import h5py
from hdfs import InsecureClient

import torch
import torch.distributed.distributed_c10d as distributed_c10d
import torch.nn
import torch.optim
from torch._C._distributed_c10d import SwiftInternalError

from .data_parallel import (_DistributedOptimizer, broadcast_optimizer_state,
                            broadcast_parameters)
from .distributed_c10d import (_failure_handler, all_gather, get_rank,
                               get_world_size)

try:
    import boto3
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _is_failure_worker(failure_workers):
    return get_rank() in failure_workers


def _set_recv_mask(client, consensus_value, pairs):
    logging_files = get_logging_files(pairs)
    while logging_files:
        files = client.ls()
        for file in files:
            if file in logging_files:
                logger.info(f"download {file} from hdfs")
                client.download(dfs_path=file, local_path=file)
                f = h5py.File(file, "r")
                keys = sorted(list(f.keys()), key=lambda x: (int(x.split(":")[0]), int(x.split(":")[1])))
                valid_keys = filter(lambda x: int(x.split(":")[0]) < consensus_value, keys)
                # (file handle, valid_keys)
                distributed_c10d._logging_recv_mask[src] = (f, valid_keys)
            logging_files.remove(file)

        time.sleep(0.1)


def run(replica=False, logging=False, *args_, **kwargs_):
    """
    kwargs:
         compression: whether to enable compression for logging data True/False
         dfs: which distributed filesystem in use ['s3', 'hfds']

    """
    distributed_c10d._logging = logging
    if logging:
        if "compression" in kwargs_:
            distributed_c10d._logging_compression = kwargs_['compression']

    def f(func):
        @functools.wraps(func)
        def wrapper(timestamp, model, optimizer, *args, **kwargs):
            assert isinstance(timestamp, Timestamp)
            assert isinstance(model, torch.nn.Module)
            assert isinstance(optimizer, torch.optim.Optimizer)
            if replica:
                assert type(optimizer).__name__ == "DistributedOptimizer"

            if distributed_c10d._logging:
                groups = get_groups(**kwargs_)
                pairs = groups_to_pairs(groups)
                if not need_logging(pairs):
                    logging = False
                    distributed_c10d._logging = False

            if logging:
                logger.info(f"enable logging on device {torch.cuda.current_device()}")
                distributed_c10d._logging_stream = torch.cuda.Stream()
                distributed_c10d._logging_cpu_tensor_queue = Queue()
                distributed_c10d._logging_dfs_client = DFSClient.create(**kwargs_)

            while True:
                try:
                    if logging:
                        distributed_c10d._logging_thread = threading.Thread(
                            target=distributed_c10d.flush_objects_to_dfs, args=(timestamp, ))
                        distributed_c10d._logging_thread.start()

                    consensus_value, need_undo, failure_workers = timestamp.sync()
                    logger.info(f"failure workers: {failure_workers}")

                    if need_undo:
                        logger.info(f"[Rank {get_rank()}] undo update is needed"
                                    "(iteration = {timestamp.value} while the consensus value is {timestamp.value-1})!")
                        optimizer.undo()

                    if replica:
                        broadcast_parameters(model.named_parameters(), 0)
                        optimizer.clear()
                        broadcast_optimizer_state(optimizer.state_dict(), 0)
                    elif logging:
                        if _is_failure_worker(failure_workers):
                            _set_recv_mask(client, consensus_value)
                        else:
                            # TODO: parallel recovery
                            pass

                    return func(timestamp, model, optimizer, *args, **kwargs)
                except SwiftInternalError as e:
                    logger.info("catch an error: " + str(e))
                    _failure_handler()
                finally:
                    if logging:
                        distributed_c10d._logging_cpu_tensor_queue.put(None)
                        distributed_c10d._logging_thread.join()

        return wrapper
    return f


class Timestamp:
    def __init__(self, value):
        if value < 0:
            raise ValueError("timestamp must not be less than zero!")
        self._value = value

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
        dfs = kwargs['dfs']
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


class S3Client(DFSClient):
    def __init__(self, *args, **kwargs):
        if "bucket" in kwargs:
            bucket = kwargs["bucket"]
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
        files = rsp['Contents']
        return [file['Key'] for file in files]


def get_groups(group_size=None, groups=None, **kwargs):
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


def need_logging(pairs):
    rank = get_rank()
    for pair in pairs:
        if rank in pair:
            return True
    return False


def get_logging_files(pairs):
    rank = get_rank()
    logging_files = []
    for pair in pairs:
        if rank in pair:
            peer = pair[1] if pair[0] == rank else pair[0]
            logging_files.append(f"logging_{peer}_{rank}.h5")
    return logging_files
