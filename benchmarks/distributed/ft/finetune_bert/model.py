from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch._C import ThroughputBenchmark
from torch.onnx.symbolic_opset9 import tensor
from torch.utils import checkpoint
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import os

import modeling
from modeling import (BertConfig, BertEmbeddings, BertForQuestionAnswering,
                      BertLayer)
from schedule import (get_microbatch_size, get_pipeline_model_parallel_rank,
                      get_pipeline_model_parallel_world_size)

# Prepare model config
config = BertConfig.from_json_file('./bert_config.json')

# Padding for divisibility by 8
if config.vocab_size % 8 != 0:
    config.vocab_size += 8 - (config.vocab_size % 8)

modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training

class QA_Outputs(nn.Module):
    def __init__(self, qa_outputs):
        super().__init__()
        self.qa_outputs = qa_outputs
    def forward(self, x):
        logits = self.qa_outputs(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

class PipelineParallelBert(nn.Module):
    def __init__(self, rank=None, balance=None, *args, **kwargs):
        super(PipelineParallelBert, self).__init__()
        self.bert = modeling.BertForQuestionAnswering.from_pretrained("bert-base-uncased",
                    cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(get_pipeline_model_parallel_rank())))

        self.bert_sequential = nn.Sequential(
            self.bert.bert.embeddings,
            *(self.bert.bert.encoder.layer),
            # it seems that is doesn't need pooler
            # self.bert.pooler,
            QA_Outputs(self.bert.qa_outputs)
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

    def _profile(self, shape=[384]):
        """
        get each layer's input/output shape by running one forward pass
        """
        micro_batch_size = get_microbatch_size()
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
                elif isinstance(layer, QA_Outputs):
                    sequence_output = input
                    start_logits, end_logits = layer(sequence_output)
                    output = [start_logits, end_logits]
                    self._output_shapes.append(len(output))
                    return
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
            elif isinstance(layer, QA_Outputs):
                sequence_output = input
                start_logits, end_logits = layer(sequence_output)
                return start_logits, end_logits

            input = output

        return output
