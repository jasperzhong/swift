import os

import torch
import torch.nn as nn

from resnet import Bottleneck, ResNet
from schedule import (get_microbatch_size, get_pipeline_model_parallel_rank,
                      get_pipeline_model_parallel_world_size)


class PipelineParallelResNet50(ResNet):
    def __init__(self, rank, balance=None, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=1000, *args, **kwargs
        )

        self.resnet50_sequential = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
            nn.Flatten(),
            self.fc
        )

        if balance is not None:
            assert len(balance) == get_pipeline_model_parallel_world_size(), \
                "The number of `balance` does not match the number of pipeline stages"
            assert sum(balance) == len(self.resnet50_sequential), \
                "The summation of `balance` does not match the number of layers"
            self.balance = balance
        else:
            num_layers_per_stage = len(self.resnet50_sequential) // \
                get_pipeline_model_parallel_world_size()
            print(f"num_layers_per_stage = {num_layers_per_stage}")
            self.balance = [num_layers_per_stage] * get_pipeline_model_parallel_world_size()
            remaining = len(self.resnet50_sequential) - num_layers_per_stage * len(self.balance)
            self.balance[-1] += remaining
            print(f"balance = {self.balance}")

        self._profile()

        self.rank = None
        self.model_split = None

        if rank is None:
            rank = get_pipeline_model_parallel_rank()
        self.assign_model_split(rank)

    def assign_model_split(self, rank):
        # assign model split
        if rank == self.rank:
            return

        self.rank = rank

        # offload previous model split from GPU
        if self.model_split is not None:
            self.model_split.cpu()

        # assign model split
        start = 0
        for i in range(rank):
            start += self.balance[i]

        end = start + self.balance[rank]
        self._input_shape = self._input_shapes[start]
        self._output_shape = self._output_shapes[end - 1]
        self.model_split = self.resnet50_sequential[start:end]

    def _profile(self, shape=[3, 224, 224]):
        """
        get each layer's input/output shape by running one forward pass
        """
        micro_batch_size = get_microbatch_size()

        if os.path.exists("profile.txt"):
            with open("profile.txt", "r") as f:
                lines = f.readlines()
            shapes = []
            self._input_shapes = None
            self._output_shapes = None
            for line in lines:
                line = line.strip('\n')
                if line:
                    nums = line.split(" ")
                    nums = [int(num) for num in nums]
                  
                    if len(nums) == 1:
                        nums = nums[0]
                    else:
                        nums = tuple(nums)
                        if nums[0] != micro_batch_size:
                            print("The microbatch size in profile.txt is not consistent. Remove profile.txt.")
                            os.remove("profile.txt")
                            break
                    shapes.append(nums)
                else:
                    self._input_shapes = shapes
                    shapes = []
            self._output_shapes = shapes
            print("read shapes from file")
            if self._input_shapes and self._output_shapes:
                return

        fake_input = torch.randn(tuple([micro_batch_size] + shape))

        self._input_shapes = []
        self._output_shapes = []
        input = fake_input
        with torch.no_grad():
            for layer in self.resnet50_sequential:
                self._input_shapes.append(input.shape)
                output = layer(input)
                self._output_shapes.append(output.shape)
                input = output
        
        local_rank = int(os.environ['LOCAL_RANK'])
        if local_rank == 0:
            with open("profile.txt", "w") as f:
                for shape in self._input_shapes:
                    if isinstance(shape, tuple):
                        f.write(' '.join(str(s) for s in shape) + '\n')
                    elif isinstance(shape, int):
                        f.write(str(shape) + '\n')
                    else:
                        raise ValueError("unrecognized type")

                f.write('\n')

                for shape in self._output_shapes:
                    if isinstance(shape, tuple):
                        f.write(' '.join(str(s) for s in shape) + '\n')
                    elif isinstance(shape, int):
                        f.write(str(shape) + '\n')
                    else:
                        raise ValueError("unrecognized type")

    def parameters(self, recurse=True):
        return self.model_split.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self.model_split.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self):
        return self.model_split.state_dict()

    def load_state_dict(self, state_dict):
        self.model_split.load_state_dict(state_dict)

    def children(self):
        if not hasattr(self, 'model_split'):
            return super(PipelineParallelResNet50, self).children()
        return self.model_split.children()

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        return self.model_split(x)
