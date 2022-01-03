import torch
import torch.nn as nn

import schedule
from schedule import (get_microbatch_size, get_pipeline_model_parallel_rank,
                      get_pipeline_model_parallel_world_size)
from vit import ViT


class PipelineParallelViT(ViT):
    def __init__(self, rank=None, balance=None, *args, **kwargs):
        super(PipelineParallelViT, self).__init__(
            image_size=224, 
            patch_size=32, 
            num_classes=1000, 
            dim=768, # base 768 ; large 1024 ; huge 1280 
            depth=12, # base 12 ; large 24 ; huge 36 
            heads=12, # base 12 ; large 16 ; huge 16
            mlp_dim=3072, # base 3072 ; large 4096 ; huge 5120
            pool = 'cls', 
            channels = 3, 
            dim_head = 64, 
            dropout = 0.1, 
            emb_dropout = 0.1
        )

        self.vit_sequential = nn.Sequential(
            nn.Sequential(
                self.embedding,
                self.dropout),
            *(self.transformer.layers),
            self.to_latent,
            self.mlp_head
            )
        
        if balance is not None:
            assert len(balance) == get_pipeline_model_parallel_world_size(), \
                "The number of `balance` does not match the number of pipeline stages"
            assert sum(balance) == len(self.vit_sequential), \
                "The summation of `balance` does not match the number of layers"
            self.balance = balance
        else:
            num_layers_per_stage = len(self.vit_sequential) // \
                get_pipeline_model_parallel_world_size()
            self.balance = [num_layers_per_stage] * get_pipeline_model_parallel_world_size()
            remaining = len(self.vit_sequential) - num_layers_per_stage * len(self.balance)
            self.balance[-1] += remaining

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
        for i in range(self.rank):
            start += self.balance[i]

        end = start + self.balance[self.rank]
        self._input_shape = self._input_shapes[start]
        self._output_shape = self._output_shapes[end - 1]
        self.model_split = self.vit_sequential[start:end]

    def _profile(self, shape=None):
        """
        get each layer's input/output shape by running one forward pass
        """
        if shape == None:
            shape = [3, schedule._GLOBAL_ARGS.img_size, schedule._GLOBAL_ARGS.img_size]
        micro_batch_size = get_microbatch_size()
        fake_input = torch.randn(tuple([micro_batch_size] + shape))

        self._input_shapes = []
        self._output_shapes = []
        input = fake_input
        with torch.no_grad():
            for layer in self.vit_sequential:
                self._input_shapes.append(input.shape)
                output = layer(input)
                self._output_shapes.append(output.shape)
                input = output

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
            return super(PipelineParallelViT, self).children()
        return self.model_split.children()


    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):      
        return self.model_split(x)
