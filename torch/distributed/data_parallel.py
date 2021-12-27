# Copyright 2019 Bytedance Inc. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import collections
import io
from contextlib import contextmanager

import cloudpickle

import torch

from .distributed_c10d import all_reduce, broadcast, get_rank, get_world_size


class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters,
                 backward_passes_per_step=1, comm_group=None):
        super(self.__class__, self).__init__(params)

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        self.backward_passes_per_step = backward_passes_per_step
        self.comm_group = comm_group
        self._all_reduce_delay = {v: self.backward_passes_per_step
                                  for _, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._should_sync = True
        self._size = get_world_size()
        if self._size > 1:
            print("DistributedOptimizer registers hooks")
            self._register_hooks()
        else:
            print("there is no need to register hooks when world size == 1")

    @staticmethod
    def find_duplicates(lst):
        seen = set()
        dups = set()
        for el in lst:
            if el in seen:
                dups.add(el)
            seen.add(el)
        return dups

    def set_backward_passes_per_step(self, passes):
        self.backward_passes_per_step = passes
        for p in self._all_reduce_delay:
            self._all_reduce_delay[p] = self.backward_passes_per_step

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _all_reduce_grad_async(self, p):
        tensor = p.grad
        tensor /= self._size
        handle = all_reduce(tensor, async_op=True, group=self.comm_group)
        return handle

    def _make_hook(self, p):
        def hook(*ignore):
            if p in self._handles and self._handles[p] is not None:
                if self._all_reduce_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            assert not p.grad.requires_grad
            assert self._all_reduce_delay[p] > 0
            handle = None
            self._all_reduce_delay[p] -= 1
            if self._all_reduce_delay[p] == 0:
                handle = self._all_reduce_grad_async(p)
            self._handles[p] = handle
        return hook

    def synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle = self._all_reduce_grad_async(p)
            self._handles[p] = handle

        for p, handle in self._handles.items():
            if handle is None:
                handle = self._all_reduce_grad_async(p)
                self._handles[p] = handle

        for p, handle in self._handles.items():
            handle.wait()
            self._all_reduce_delay[p] = self.backward_passes_per_step
        self._handles.clear()

    def clear(self):
        for p, handle in self._handles.items():
            handle.wait()
            self._all_reduce_delay[p] = self.backward_passes_per_step
        self._handles.clear()
        self.set_backward_passes_per_step(self.backward_passes_per_step)

    @contextmanager
    def skip_synchronize(self):
        self._should_sync = False
        try:
            yield
        finally:
            self._should_sync = True

    def step(self, closure=None):
        # skip sync if calling skip_synchronize
        if self._should_sync:
            self.synchronize()
        return super(self.__class__, self).step(closure)

    def undo(self):
        return super(self.__class__, self).undo()

    def state_dict(self):
        return super(self.__class__, self).state_dict()


def DistributedOptimizer(optimizer, named_parameters=None,
                         backward_passes_per_step=1, comm_group=None):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an all_reduce to
    average gradient values before applying gradients to model weights.
    all_reduce operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all all_reduce operations are
    finished before applying gradients to the model.
    DistributedOptimizer exposes the `synchronize()` method, which forces all_reduce operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.
    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```
    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          all_reduce operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during all_reduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
        backward_passes_per_step: Number of expected backward passes to perform
                                  before calling step()/synchronize(). This
                                  allows accumulating gradients over multiple
                                  mini-batches before executing averaging and
                                  applying them.
        comm_group: communication group 
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an all_reduce implementation.
    cls = type("DistributedOptimizer", (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters,
               backward_passes_per_step, comm_group)


def broadcast_parameters(params, root_rank, comm_group=None):
    """
      Broadcasts the parameters from root rank to all other processes.
      Typical usage is to broadcast the `model.state_dict()`,
      `model.named_parameters()`, or `model.parameters()`.
      Arguments:
          params: One of the following:
              - list of parameters to broadcast
              - dict of parameters to broadcast
          root_rank: The rank of the process from which parameters will be
                     broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run synchronous broadcasts.
    for name, p in params:
        broadcast(p, root_rank, group=comm_group)


def broadcast_optimizer_state(optimizer, root_rank, prefix="Parameter.", comm_group=None):
    """
    Broadcasts an optimizer state from root rank to all other processes.
    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces all_reduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    scalars = {}
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we place the scalars into a single dict,
    # then pickle and broadcast with broadcast_object (under the assumption
    # that there are not many scalars, and so the overhead of pickling will
    # be relatively low). Because broadcast_object is performed out-of-place,
    # we then use a callback to assign the new value to the correct element
    # of the optimizer state.
    def _create_state_callback(pid, name):
        def _assign_state(v):
            state_dict['state'][pid][name] = v
        return _assign_state

    def _create_option_callback(index, option_key):
        def _assign_option(v):
            optimizer.param_groups[index][option_key] = v
        return _assign_option

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be broadcast separately
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value]).cuda()
            scalars[key] = option_value
            callbacks[key] = _create_option_callback(index, option_key)

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            if pid not in state_dict['state']:
                # The param has not set requires_grad, so skip broadcast
                continue

            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if torch.is_tensor(p):
                    # Tensor -> use broadcast_parameters
                    params.append((key, p))
                else:
                    # Scalar -> use broadcast_object
                    scalars[key] = p
                    callbacks[key] = _create_state_callback(pid, name)

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank, comm_group)

    # Broadcast and cleanup for non-tensor parameters
    scalars = broadcast_object(scalars, root_rank, comm_group=comm_group)
    for key, p in scalars.items():
        callbacks[key](p)


def broadcast_object(obj, root_rank=0, name=None, comm_group=None):
    """
    Serializes and broadcasts an object from root rank to all other processes.
    Typical usage is to broadcast the `optimizer.state_dict()`, for example:
    .. code-block:: python
        state_dict = broadcast_object(optimizer.state_dict(), 0)
        if get_rank() > 0:
            optimizer.load_state_dict(state_dict)
    Arguments:
        obj: An object capable of being serialized without losing any context.
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
        name: Optional name to use during broadcast, will default to the class
              type.
    Returns:
        The object that was broadcast from the `root_rank`.
    """
    if name is None:
        name = type(obj).__name__

    if get_rank() == root_rank:
        b = io.BytesIO()
        cloudpickle.dump(obj, b)
        t = torch.ByteTensor(bytearray(b.getvalue())).cuda()
        sz = torch.IntTensor([t.shape[0]]).cuda()
        broadcast_parameters([(name + '.sz', sz)], root_rank, comm_group)
    else:
        sz = torch.IntTensor([0]).cuda()
        broadcast_parameters([(name + '.sz', sz)], root_rank, comm_group)
        t = torch.ByteTensor(sz.cpu().tolist()[0]).cuda()

    broadcast_parameters([(name + '.t', t)], root_rank, comm_group)

    if get_rank() != root_rank:
        buf = io.BytesIO(t.cpu().numpy().tobytes())
        obj = cloudpickle.load(buf)

    return obj
