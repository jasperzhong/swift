import os
import tempfile
import copy
import warnings
import inspect
import re

import torch
from torch._six import PY2
import torch.testing._internal.common_utils as common
import torch.testing._internal.common_nn as common_nn
import torch.utils.cpp_extension
from cpp_api_parity.parity_table_parser import parse_parity_tracker_table
from cpp_api_parity import functional_impl_check_v2 as functional_impl_check
from cpp_api_parity import module_impl_check

# yf225 TODO: need to add proper checks and expectations when people:
# 1. Add a new test to a module already supported by C++ API (i.e. parity table has entry for it, and the parity bit is yes)
#   a) add a flag `no_cpp_parity_test` to the dict to be able to turn off test as needed
# 2. Add a new test for a module that is not supported by C++ API yet

# yf225 TODO: our new parity test mechanism is changing the way people write common_nn test dicts a lot...
# can we enforce some constraints to make writing common_nn test dicts easier / get less questions from people?

# yf225 TODO: current plan:
# transfer all input/target/extra/options args from Python to C++ via JIT serialization
# transfer module state from Python to C++ via JIT tracing (torch.jit.save -> torch::load)
# Benefits:
# 1. Allows minimal changes to common_nn dicts
# 2. Easy to maintain common_nn dicts, no tricky things to watch out for

class TestCppApiParity(common.TestCase):
  pass

parity_table_path = os.path.join(os.path.dirname(__file__), 'cpp_api_parity/parity-tracker.md')

parity_table = parse_parity_tracker_table(parity_table_path)

# module_tests = common_nn.module_tests
# new_module_tests = common_nn.new_module_tests
# criterion_tests = common_nn.criterion_tests
# new_criterion_tests = common_nn.new_criterion_tests

module_tests = []
new_module_tests = []
criterion_tests = []
new_criterion_tests = []

import torch.nn.functional as F

def wrap_functional(fn, **kwargs):
    class FunctionalModule(torch.nn.Module):
        def forward(self, *args):
            return fn(*args, **kwargs)
    return FunctionalModule

def bceloss_weights_no_reduce_scalar_test():
    t = torch.randn(()).double()
    weights = torch.rand(())
    return dict(
        fullname='BCELoss_weights_no_reduce_scalar',
        functional_name='binary_cross_entropy',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i),
                                             weight=weights.type_as(i), reduction='none')),
        cpp_constructor='F::binary_cross_entropy(i, t.to(i.options()), F::BinaryCrossEntropyFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone)',
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        # RHS value format: 'input' / 'target' / 'extra_args_0' / 'extra_args_1'
        # NOTE: must be full binding (i.e. all input args to the functional need to be specified)
        cpp_args={'i': 'input', 't': t, 'weights': weights},
        reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()) * weights,
        pickle=False
    )

def fractional_max_pool2d_test(test_case):
    random_samples = torch.DoubleTensor(1, 3, 2).uniform_()
    return dict(
        constructor=lambda: torch.nn.FractionalMaxPool2d(
            2, output_ratio=0.5, _random_samples=random_samples),
        cpp_constructor_args='torch::nn::FractionalMaxPool2dOptions(2).output_ratio(0.5)._random_samples(random_samples)',
        input_size=(1, 3, 5, 7),
        # RHS value format: 'input' / 'target' / 'extra_args_0' / 'extra_args_1'
        # NOTE: must be full binding (i.e. all input args to the module forward need to be specified)
        cpp_args={'i': 'input', 'random_samples', random_samples},
        fullname='FractionalMaxPool2d_ratio')

new_module_tests.append(bceloss_weights_no_reduce_scalar_test())
new_module_tests.append(fractional_max_pool2d_test())

for test_params_dicts, test_instance_class in [
  (module_tests, common_nn.ModuleTest),
  (new_module_tests, common_nn.NewModuleTest),
  (criterion_tests, common_nn.CriterionTest),
  (new_criterion_tests, common_nn.NewCriterionTest),
]:
  module_impl_check.add_tests(TestCppApiParity, test_params_dicts, test_instance_class, parity_table)
  functional_impl_check.add_tests(TestCppApiParity, test_params_dicts, test_instance_class, parity_table)

module_impl_check.build_cpp_tests(TestCppApiParity)
functional_impl_check.build_cpp_tests(TestCppApiParity)

if __name__ == "__main__":
  common.run_tests()
