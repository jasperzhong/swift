import functools
import os
import re
import threading
from queue import Queue

import torch
import torch.distributed.distributed_c10d as distributed_c10d
import torch.nn
import torch.optim
from torch._C._distributed_c10d import SwiftInternalError

from .data_parallel import (_DistributedOptimizer, broadcast_optimizer_state,
                            broadcast_parameters)
from .distributed_c10d import (_failure_handler, all_gather, get_rank,
                               get_world_size)


def _is_failure_worker(failure_workers):
    return get_rank() in failure_workers


def _need_to_resend(failure_workers):
    logging_files = [f for f in os.listdir('.') if re.match("logging_.*\.h5", f)]
    for f in logging_files:
        _, src, dst = f.split('.')[0].split("_")
        src, dst = int(src), int(dst)
        if src == get_rank() and dst in failure_workers:
            return True
    return False


def _resend(failure_workers):
    pass


def run(replica=False, logging=False, compression=None):
    distributed_c10d._logging = logging
    distributed_c10d._logging_compression = compression

    def f(func):
        @functools.wraps(func)
        def wrapper(timestamp, model, optimizer, *args, **kwargs):
            assert isinstance(timestamp, Timestamp)
            assert isinstance(model, torch.nn.Module)
            assert isinstance(optimizer, torch.optim.Optimizer)
            if replica:
                assert type(optimizer).__name__ == "DistributedOptimizer"

            if logging:
                print(f"enable logging on device {torch.cuda.current_device()}")
                distributed_c10d._logging_stream = torch.cuda.Stream()
                distributed_c10d._logging_cpu_tensor_queue = Queue()
                distributed_c10d._logging_thread = threading.Thread(target=distributed_c10d.flush_objects_to_fs)
                distributed_c10d._logging_thread.start()

            while True:
                try:
                    need_undo, failure_workers = timestamp.sync()
                    print("failure workers: ", failure_workers)

                    if need_undo:
                        print(f"[Rank {get_rank()}] undo update is needed"
                              "(iteration = {timestamp.value} while the consensus value is {timestamp.value-1})!")
                        optimizer.undo()

                    if replica:
                        broadcast_parameters(model.named_parameters(), 0)
                        optimizer.clear()
                        broadcast_optimizer_state(optimizer.state_dict(), 0)
                    elif logging:
                        if _is_failure_worker(failure_workers):
                            # (TODO): load checkpoint
                            pass
                        elif _need_to_resend(failure_workers):
                            print(f"[Rank {get_rank()} needs to resend logging data]")
                            _resend(failure_workers)

                    func(timestamp, model, optimizer, *args, **kwargs)

                    if logging:
                        distributed_c10d._logging_cpu_tensor_queue.put(None)
                        distributed_c10d._logging_thread.join()

                    return
                except SwiftInternalError as e:
                    print("catch an error: " + str(e))
                    _failure_handler()
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
        return self._value

    def __sub__(self, other):
        return self._value - other

    def __isub__(self, other):
        self._value -= other
        return self._value

    def __mul__(self, other):
        return self._value * other

    def __imul__(self, other):
        self._value *= other
        return self._value

    def __div__(self, other):
        return self._value / other

    def __idiv__(self, other):
        self._value /= other
        return self._value

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
        print("values:", values)

        need_undo = False
        failure_workers = []
        # all start from 0
        if not any(values):
            return need_undo, failure_workers

        for i, v in enumerate(values):
            if v == 0:
                failure_workers.append(i)

        if self._value == 0:
            # failure workers
            return need_undo, failure_workers
        else:
            # living workers
            consensus_value = self._make_consensus(values)
            self = Timestamp(consensus_value)
            need_undo = consensus_value == self._value - 1
            return need_undo, failure_workers

    def _make_consensus(self, values):
        max_v = max(values)
        if max_v - 1 in values:
            return max_v - 1
        return max_v
