import os

import torch
import torch.nn as nn
import timm.models.vision_transformer as models

from schedule import (get_microbatch_size, get_pipeline_model_parallel_rank,
                      get_pipeline_model_parallel_world_size)


class Tokens(nn.Module):
    def __init__(self, cls_token, dist_token):
        super().__init__()
        self.cls_token = cls_token
        self.dist_token = dist_token

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        return x


class PosEmb(nn.Module):
    def __init__(self, pos_embed, pos_drop):
        super().__init__()
        self.pos_embed = pos_embed
        self.pos_drop = pos_drop

    def forward(self, x):
        return self.pos_drop(x + self.pos_embed)


class Cls(nn.Module):
    def __init__(self, pre_logits, head):
        super().__init__()
        self.pre_logits = pre_logits
        self.head = head

    def forward(self, x):
        x = self.pre_logits(x[:, 0])
        x = self.head(x)
        return x


class Norm(nn.Module):
    def __init__(self, flatten, norm):
        super().__init__()
        self.flatten = flatten
        self.norm = norm

    def forward(self, x):
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PipelineParallelViT(nn.Module):
    def __init__(self, rank=None, balance=None, *args, **kwargs):
        super(PipelineParallelViT, self).__init__()
        # model_kwargs = dict(
        # patch_size=32, embed_dim=1024, depth=126, num_heads=16, representation_size=1024, img_size=224)
        # self.vit = models._create_vision_transformer('vit_large_patch32_224_in21k', pretrained=False, **model_kwargs)
        # model_kwargs = dict(
        # patch_size=14, embed_dim=1280, depth=126, num_heads=16, representation_size=1280, img_size=224)
        # self.vit = models._create_vision_transformer('vit_huge_patch14_224_in21k', pretrained=False, **model_kwargs)
        model_kwargs = dict(
            patch_size=32, embed_dim=768, depth=126, num_heads=12, num_classes=1000, img_size=224)
        self.vit = models._create_vision_transformer("vit_base_patch32_224_in21k", pretrained=False,
                                                     **model_kwargs)
        self.vit_sequential = nn.Sequential(
            nn.Sequential(
                self.vit.patch_embed.proj,
                Norm(
                    self.vit.patch_embed.flatten,
                    self.vit.patch_embed.norm
                ),
                Tokens(
                    self.vit.cls_token,
                    self.vit.dist_token),
                PosEmb(
                    self.vit.pos_embed,
                    self.vit.pos_drop)),
            *(self.vit.blocks),
            self.vit.norm,
            Cls(
                self.vit.pre_logits,
                self.vit.head
            )
        )

        if balance is not None:
            assert len(balance) == get_pipeline_model_parallel_world_size(), \
                "The number of `balance` {}does not match the number of pipeline stages".format(len(balance))
            assert sum(balance) == len(self.vit_sequential), \
                "The summation of `balance` {} does not match the number of layers {}".format(
                    sum(balance), len(self.vit_sequential))
            self.balance = balance
        else:
            print("vit seq len: {}".format(len(self.vit_sequential)))
            num_layers_per_stage = len(self.vit_sequential) // \
                get_pipeline_model_parallel_world_size()
            self.balance = [num_layers_per_stage] * get_pipeline_model_parallel_world_size()
            print("num_layers_per_stage: {}".format(num_layers_per_stage))
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

        if shape is None:
            shape = [3, 224, 224]
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
