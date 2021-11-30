import hashlib
import functools

import torch
import torch.nn
import torch.optim
from torch._C._distributed_c10d import SwiftInternalError

from .data_parallel import (_DistributedOptimizer, broadcast_optimizer_state,
                            broadcast_parameters)
from .distributed_c10d import _failure_handler, all_gather, get_rank, get_world_size


def run(func):
    @functools.wraps(func)
    def wrapper(state, *args, **kwargs):
        while True:
            try:
                if state:
                    state.sync()

                return func(state, *args, **kwargs)
            except SwiftInternalError as e:
                print("catch an error: " + str(e))
                _failure_handler()
    return wrapper


class _AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(_AttrDict, self).__init__(*args, dict(sorted(kwargs.items())))
        self.__dict__ = self


class State(_AttrDict):
    def __init__(self, *args, **kwargs):
        super(State, self).__init__(*args, **kwargs)

    def sync(self):
        need_undo = False
        for k, v in self.items():
            if isinstance(v, torch.nn.Module):
                self._model_parameters_sync_handler(k, v)
            elif isinstance(v, torch.optim.Optimizer):
                self._optimizer_state_sync_handler(k, v)
            elif isinstance(v, int):
                ret = self._timestamp_sync_handler(k, v)
                if ret:
                    print(f"[Rank {get_rank()}] undo update is needed ({k} = {v} while the consensus value is {v-1})!")
                    need_undo = True
            else:
                raise ValueError(f"type {type(v)} of key({k}) is not recognized!")

        if need_undo:
            for k, v in self.items():
                if isinstance(v, torch.optim.Optimizer):
                    v.undo()

    def _model_parameters_sync_handler(self, k, v):
        broadcast_parameters(v.state_dict(), 0)

    def _optimizer_state_sync_handler(self, k, v):
        if type(v).__name__ == "DistributedOptimizer":
            v.clear()
            broadcast_optimizer_state(v, 0)

    def _timestamp_sync_handler(self, k, v):
        tensor = torch.LongTensor(3).cuda()
        tensor_list = [torch.LongTensor(3).cuda() for _ in range(get_world_size())]
        md5 = hashlib.md5(k.encode('utf-8'))
        tensor[0] = int.from_bytes(md5.digest()[:8], byteorder='little', signed=True)
        tensor[1] = int.from_bytes(md5.digest()[8:], byteorder='little', signed=True)
        tensor[2] = v
        all_gather(tensor_list, tensor)

        values = []
        for t in tensor_list:
            if t[0] != tensor[0] or t[1] != tensor[1]:
                raise RuntimeError("timestamp key not matched!")
            values.append(int(t[2].item()))

        consensus_value = self._make_consensus(values)
        self[k] = consensus_value
        return consensus_value == v - 1

    def _make_consensus(self, values):
        max_v = max(values)
        if max_v - 1 in values:
            return max_v - 1
        return max_v
