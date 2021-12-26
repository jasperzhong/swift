import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet

from schedule import get_microbatch_size, get_pipeline_model_parallel_rank, \
    get_pipeline_model_parallel_world_size


class PipelineParallelResNet50(ResNet):
    def __init__(self, balance=None, *args, **kwargs):
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
            self.balance = [num_layers_per_stage] * get_pipeline_model_parallel_world_size()
            remaining = len(self.resnet50_sequential) - num_layers_per_stage * len(self.balance)
            self.balance[-1] += remaining

        self._profile()

        self.rank = get_pipeline_model_parallel_rank()

        # assign model split
        start = 0
        for i in range(self.rank):
            start += self.balance[i]

        end = start + self.balance[self.rank]
        self._input_shape = self._input_shapes[start]
        self._output_shape = self._output_shapes[end - 1]
        self.model_split = self.resnet50_sequential[start:end]

    def _profile(self, shape=[3, 224, 224]):
        """
        get each layer's input/output shape by running one forward pass
        """
        micro_batch_size = get_microbatch_size()
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

    def parameters(self, recurse=True):
        return self.model_split.parameters(recurse=recurse)

    def state_dict(self):
        return self.model_split.state_dict()

    def load_state_dict(self, state_dict):
        self.model_split.load_state_dict(state_dict)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        return self.model_split(x)
