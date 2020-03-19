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
from cpp_api_parity import functional_impl_check, module_impl_check

# yf225 TODO: need to add proper checks and expectations when people:
# 1. Add a new test to a module already supported by C++ API (i.e. parity table has entry for it, and the parity bit is yes)
# 2. Add a new test for a module that is not supported by C++ API yet

# yf225 TODO: our new parity test mechanism is changing the way people write common_nn test dicts a lot...
# can we enforce some constraints to make writing common_nn test dicts easier / get less questions from people?

# yf225 TODO: 
# Step 1: add test to test_nn.py to enforce that argument values *change* with different manual seed,
# (aka. no predefined random constant tensor is allowed! becuase we don't have control of RNG state when generating that random tensor)
# ```
# with freeze_rng_state():
#   torch.manual_seed(0)
#   constructor_args = self.constructor_args()
#   input = self._get_input()
#   target = self._get_target() # only for criterion test
#   extra_args = self.extra_args()
#
#   torch.manual_seed(10)
#   assert constructor_args != self.constructor_args() # NOTE: check this for every arg in constructor_args
#   assert input != self._get_input()
#   assert target != self._get_target() # only for criterion test
#   assert extra_args != self.extra_args() # NOTE: check this for every arg in constructor_args
# ```
#
# Step 2: ban the use of `constructor=`, and always use `constructor_args_fn` to pass constructor_args


class TestCppApiParity(common.TestCase):
  pass

parity_table_path = os.path.join(os.path.dirname(__file__), 'cpp_api_parity/parity-tracker.md')

parity_table = parse_parity_tracker_table(parity_table_path)

module_tests = common_nn.module_tests
new_module_tests = common_nn.new_module_tests
criterion_tests = common_nn.criterion_tests
new_criterion_tests = common_nn.new_criterion_tests

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
