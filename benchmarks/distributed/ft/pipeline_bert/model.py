import torch
from torch._C import ThroughputBenchmark
import torch.nn as nn
import modeling
from modeling import BertForPreTraining, BertConfig, BertEmbeddings, BertLayer, BertPooler, BertPreTrainingHeads, BertLayerNorm
from typing import Optional, Iterable
from schedule import get_microbatch_size, get_pipeline_model_parallel_rank, \
    get_pipeline_model_parallel_world_size, is_pipeline_first_stage
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
        else:
            self.rank = rank

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
                    output = layer(hidden_states=hidden_states , attention_mask=extended_attention_mask)
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
                    return
                self._output_shapes.append(output.shape)
                input = output
            
            
    def parameters(self, recurse=True):
        return self.model_split.parameters(recurse=recurse)
        
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
                output = layer(hidden_states=hidden_states , attention_mask=extended_attention_mask)
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
