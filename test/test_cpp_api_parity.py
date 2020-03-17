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
from cpp_api_parity import sample_module, torch_nn_modules
from cpp_api_parity import functional_impl_check, module_impl_check

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
  module_impl_check.add_tests(TestCppApiParity, test_params_dicts, test_instance_class, torch_nn_modules, parity_table)
  # functional_impl_check.add_tests(module_tests, criterion_tests, torch_nn_modules, parity_table)

module_impl_check.build_cpp_tests(TestCppApiParity)

if __name__ == "__main__":
  common.run_tests()
