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
from cpp_api_parity import module_impl_check, functional_impl_check, sample_module, sample_functional

print_cpp_source = True

# yf225 TODO: need to add proper checks and expectations when people:
# 1. Add a new test to a module already supported by C++ API (i.e. parity table has entry for it, and the parity bit is yes)
#   a) add a flag `test_cpp_api_parity` to the dict to be able to turn off test as needed
# 2. Add a new test for a module that is not supported by C++ API yet

class TestCppApiParity(common.TestCase):
  pass

parity_table_path = os.path.join(os.path.dirname(__file__), 'cpp_api_parity/parity-tracker.md')

parity_table = parse_parity_tracker_table(parity_table_path)

import torch.nn.functional as F
from torch.testing._internal.common_nn import wrap_functional

# How to get the functional name from wrap_functional:
# ```
# def wrap_functional(fn, **kwargs):
#     class FunctionalModule(torch.nn.Module):
#         def __init__(self):
#             self.fn = fn
#         def forward(self, *args):
#             return self.fn(*args, **kwargs)
#     return FunctionalModule
#
# func_dict = dict(
#         functional_name='interpolate',
#         constructor=wrap_functional(F.interpolate, size=(12, ), scale_factor=None, mode='nearest'),
#         input_size=(1, 2, 3),
#         fullname='interpolate_nearest_tuple_1d',
#         pickle=False,
#     )
#
# import inspect
# print(func_dict['constructor']().fn)  # prints: <function interpolate at 0x7f13611a0ea0>
# ```

# yf225 TODO comment:
# RHS value format: 'input' / 'target' / 'extra_args_0' / 'extra_args_1'
# NOTE: any var symbol written in the cpp_* fields needs to have a mapping here!

def bceloss_weights_no_reduce_scalar_test():
    t = torch.randn(()).double()
    weights = torch.rand(())
    return dict(
        fullname='BCELoss_weights_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i),
                                             weight=weights.type_as(i), reduction='none')),
        cpp_function_call='F::binary_cross_entropy(i, t.to(i.options()), F::BinaryCrossEntropyFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone))',
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        # RHS value format: 'input' / 'target' / 'extra_args_0' / 'extra_args_1'
        # NOTE: any var symbol written in the cpp_* fields needs to have a mapping here!
        cpp_arg_symbol_map={'i': 'input', 't': t, 'weights': weights},
        reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()) * weights,
        pickle=False,
        test_cpp_api_parity=False, # yf225 TODO: this is an optional flag
    )

def interpolate_nearest_tuple_1d():
    return dict(
        constructor=wrap_functional(F.interpolate, size=(12, ), scale_factor=None, mode='nearest'),
        cpp_options_args='F::InterpolateFuncOptions().size(std::vector<int64_t>({12})).scale_factor(c10::nullopt).mode(torch::kNearest)',
        input_size=(1, 2, 3),
        fullname='interpolate_nearest_tuple_1d',
        pickle=False,
    )

def fractional_max_pool2d_test():
    random_samples = torch.empty(1, 3, 2).uniform_()
    return dict(
        constructor=lambda: torch.nn.FractionalMaxPool2d(
            2, output_ratio=0.5, _random_samples=random_samples),
        cpp_constructor_args='torch::nn::FractionalMaxPool2dOptions(2).output_ratio(0.5)._random_samples(random_samples)',
        input_size=(1, 3, 5, 7),
        # RHS value format: 'input' / 'target' / 'extra_args_0' / 'extra_args_1'
        # NOTE: any var symbol written in the cpp_* fields needs to have a mapping here!
        cpp_arg_symbol_map={'random_samples': random_samples},
        fullname='FractionalMaxPool2d_ratio')

def BCELoss_test():
    return dict(
        module_name='BCELoss',
        input_fn=lambda: torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(15, 10).gt(0).to(torch.get_default_dtype()),
        reference_fn=lambda i, t, m: -(t * i.log() + (1 - t) * (1 - i).log()).sum() /
            (i.numel() if get_reduction(m) else 1),
        check_gradgrad=False,
        check_bfloat16=False,
    )

module_tests = common_nn.module_tests
new_module_tests = common_nn.new_module_tests
criterion_tests = common_nn.criterion_tests
new_criterion_tests = common_nn.new_criterion_tests

# module_tests = []
# new_module_tests = []
# criterion_tests = []
# new_criterion_tests = []

# Functional
# new_module_tests.append(bceloss_weights_no_reduce_scalar_test())
# new_module_tests.append(interpolate_nearest_tuple_1d())

# Module
# new_module_tests.append(fractional_max_pool2d_test())
# criterion_tests.append(BCELoss_test())

for test_params_dicts, test_instance_class in [
  (sample_module.module_tests, common_nn.ModuleTest),
  (sample_functional.functional_tests, common_nn.NewModuleTest),
  (module_tests, common_nn.ModuleTest),
  (new_module_tests, common_nn.NewModuleTest),
  (criterion_tests, common_nn.CriterionTest),
  (new_criterion_tests, common_nn.NewCriterionTest),
]:
  module_impl_check.add_tests(TestCppApiParity, test_params_dicts, test_instance_class, parity_table)
  functional_impl_check.add_tests(TestCppApiParity, test_params_dicts, test_instance_class, parity_table)

module_impl_check.build_cpp_tests(TestCppApiParity, print_cpp_source=print_cpp_source)
functional_impl_check.build_cpp_tests(TestCppApiParity, print_cpp_source=print_cpp_source)

if __name__ == "__main__":
  common.run_tests()
