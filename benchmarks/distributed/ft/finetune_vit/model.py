import torch
from torch._C import ThroughputBenchmark
import torch.nn as nn
from typing import Optional, Iterable
from timm.models import create_models
from schedule import get_microbatch_size, get_pipeline_model_parallel_rank, \
    get_pipeline_model_parallel_world_size, is_pipeline_first_stage
from torch.onnx.symbolic_opset9 import tensor
from torch.utils import checkpoint

class Embeddings(nn.Module):
    def __init__(self, patch_embed, cls_token, dist_token, pos_embed, pos_drop):
        self.patch_embed = patch_embed
        self.cls_token = cls_token
        self.dist_token = dist_token
        self.pos_embed = pos_embed
        self.pos_drop = pos_drop
    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x

class Cls(nn.Module):
    def __init__(self, pre_logits, head):
        self.pre_logits = pre_logits
        self.head = head
    def forward(self ,x):
        x = self.pre_logits(x[:, 0])
        x = self.head(x)
        return x

class PipelineParallelViT(nn.Module):
    def __init__(self, rank=None, balance=None, *args, **kwargs):
        super(PipelineParallelViT, self).__init__()
        # patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280
        self.vit = create_models("vit_huge_patch14_224_in21k", pretrain=True, num_classes=100)
        self.vit_sequential = nn.Sequential(
            Embeddings(
                self.vit.patch_embed,
                self.vit.cls_token,
                self.vit.dist_token,
                self.vit.pos_embed,
                self.vit.pos_drop),
            *(self.vit.blocks),
            self.vit.norm,
            Cls(
                self.vit.pre_logits,
                self.vit.head
                )
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

        if rank is None:
            self.rank = get_pipeline_model_parallel_rank()
        else:
            self.rank = rank

        # assign model split
        start = 0
        for i in range(self.rank):
            start += self.balance[i]

        end = start + self.balance[self.rank]
        self._input_shape = self._input_shapes[start]
        self._output_shape = self._output_shapes[end - 1]
        self.model_split = self.vit_sequential[start:end]

    def _profile(self, shape=[3, 384, 384]):
        """
        get each layer's input/output shape by running one forward pass
        """
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
