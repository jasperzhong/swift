import torch
import torch.nn as nn
import modeling
from modeling import BertForPreTraining, BertForMaskedLM, BertConfig
from typing import Optional, Iterable
from schedule import get_microbatch_size, get_pipeline_model_parallel_rank, \
    get_pipeline_model_parallel_world_size, is_pipeline_first_stage
from torch.nn.modules.container import Sequential

# Prepare model config
config = BertConfig.from_json_file('./bert_config.json')

# Padding for divisibility by 8
if config.vocab_size % 8 != 0:
    config.vocab_size += 8 - (config.vocab_size % 8)

modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training

class PipelineParallelBert(BertForPreTraining):
    def __init__(self, rank=None, balance=None, *args, **kwargs):
        super(PipelineParallelBert, self).__init__(
            config=config
        )

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

        if rank is None:
            self.rank = get_pipeline_model_parallel_rank()

        # assign model split
        start = 0
        for i in range(self.rank):
            start += self.balance[i]

        end = start + self.balance[self.rank]
        self._input_shape = self._input_shapes[start]
        self._output_shape = self._output_shapes[end - 1]
        self.model_split = self.bert[start:end]

    def _profile(self, shape=[128]):
        """
        get each layer's input/output shape by running one forward pass
        """
        micro_batch_size = get_microbatch_size()
        fake_input = torch.randn(tuple([micro_batch_size] + shape))

        self._input_shapes = []
        self._output_shapes = []
        input = fake_input
        with torch.no_grad():
            for layer in self.bert_sequential:
                self._input_shapes.append(input.shape)
                output = layer(input)
                self._output_shapes.append(output.shape)
                input = output

    def parameters(self, recursive=True):
        params = []
        for layer in self.model_split:
            params.extend(list(layer.parameters()))

        for param in params:
            yield param

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        if is_pipeline_first_stage():
            input_ids, segment_ids, input_mask = x
            return self.model_split(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        return self.model_split(x)