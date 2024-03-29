import os
import time
from typing import Iterable, Optional

import modeling
from modeling import (BertConfig, BertEmbeddings, BertForPreTraining,
                      BertLayer, BertPooler, BertPreTrainingHeads)
from schedule import (get_microbatch_size, get_pipeline_model_parallel_rank,
                      get_pipeline_model_parallel_world_size)

import torch
import torch.nn as nn
from torch._C import ThroughputBenchmark
from torch.onnx.symbolic_opset9 import tensor
from torch.utils import checkpoint

# Prepare model config
config = BertConfig.from_json_file('./bert_config.json')

# Padding for divisibility by 8
if config.vocab_size % 8 != 0:
    config.vocab_size += 8 - (config.vocab_size % 8)

modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training


class PipelineParallelBert(BertForPreTraining):
    def __init__(self, rank=None, balance=None, *args, **kwargs):
        start = time.time()
        super(PipelineParallelBert, self).__init__(
            config=config
        )
        print(f"super PipelineParallelBert {time.time() - start}")

        self.bert_sequential = nn.Sequential(
            self.bert.embeddings,
            *(self.bert.encoder.layer),
            self.bert.pooler,
            self.cls
        )

        if balance is not None:
            assert len(balance) == get_pipeline_model_parallel_world_size(), \
                "The number of `balance` does not match the number of pipeline stages"
            assert sum(balance) == len(self.bert_sequential), \
                "The summation of `balance` does not match the number of layers"
            self.balance = balance
        else:
            num_layers_per_stage = len(self.bert_sequential) // \
                get_pipeline_model_parallel_world_size()
            self.balance = [num_layers_per_stage] * get_pipeline_model_parallel_world_size()
            remaining = len(self.bert_sequential) - num_layers_per_stage * len(self.balance)
            self.balance[-1] += remaining

        self._profile()

        self.rank = None
        self.model_split = None

        if rank is None:
            rank = get_pipeline_model_parallel_rank()
        self.assign_model_split(rank)

        start = time.time()
        self.apply(self.init_bert_weights)
        print(f"Apply init_bert_weights {time.time() - start}")

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
        self.model_split = self.bert_sequential[start:end]

    def _profile(self, shape=[128]):
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

        fake_input_ids = torch.randint(1, 2, tuple([micro_batch_size] + shape))
        fake_segment_ids = torch.randint(1, 2, tuple([micro_batch_size] + shape))
        fake_input_mask = torch.randint(1, 2, tuple([micro_batch_size] + shape))
        extended_attention_mask = fake_input_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        self._input_shapes = []
        self._output_shapes = []
        input = fake_input_ids
        with torch.no_grad():
            for layer in self.bert_sequential:
                self._input_shapes.append(input.shape if isinstance(input, torch.Tensor) else len(input))
                if isinstance(layer, BertEmbeddings):
                    output = layer(input_ids=fake_input_ids, token_type_ids=fake_segment_ids)
                elif isinstance(layer, BertLayer):
                    hidden_states = input
                    output = layer(hidden_states=hidden_states, attention_mask=extended_attention_mask)
                elif isinstance(layer, BertPooler):
                    encoded_layers = input
                    pooled_output = layer(hidden_states=encoded_layers)
                    output = [encoded_layers, pooled_output]
                    self._output_shapes.append(len(output))
                    input = output
                    continue
                elif isinstance(layer, BertPreTrainingHeads):
                    encoded_layers, pooled_output = input
                    output = layer(encoded_layers, pooled_output)
                    self._output_shapes.append(len(output))
                    break
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
            return super(PipelineParallelBert, self).children()
        return self.model_split.children()

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, input, token_type_ids, attention_mask):
        for layer in self.model_split:
            if isinstance(layer, BertEmbeddings):
                input_ids = input
                output = layer(input_ids=input_ids, token_type_ids=token_type_ids)
            elif isinstance(layer, BertLayer):
                hidden_states = input
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and -10000.0 for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                output = layer(hidden_states=hidden_states, attention_mask=extended_attention_mask)
            elif isinstance(layer, BertPooler):
                encoded_layers = input
                pooled_output = layer(hidden_states=encoded_layers)
                output = [encoded_layers, pooled_output]
            elif isinstance(layer, BertPreTrainingHeads):
                encoded_layers, pooled_output = input
                output = layer(encoded_layers, pooled_output)
                return output
            input = output
        return output
