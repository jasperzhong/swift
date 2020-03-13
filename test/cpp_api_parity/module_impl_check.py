import tempfile
import shutil
from string import Template
import unittest
from torch.testing._internal.common_cuda import TEST_CUDA

import torch
import torch.testing._internal.common_nn as common_nn
from cpp_api_parity.utils import TorchNNTestParams
from cpp_api_parity import torch_nn_modules

# DO NOW:
# yf225 TODO: can we make things significantly simpler to understand, even if that means doing some manual work to every test dict?
# TODO: we should write the CPP ctor args and input args DIRECTLY in the test dict in common_nn, in order to significant simplify logic in this file

# Check 1: Module implementation correctness check:

# Step 1: Translate ctor args from Python layer to C++ layer
# Step 2: Construct a C++ layer, run forward and backward on it, save all its params/buffers/gradients into a ScriptModule
# Step 3: Load that ScriptModule into Python, and compare output/params/buffers/gradients with Python layer (forward and backward)

# yf225 TODO: move to common utils?
# devices = ['cpu', 'cuda']
devices = ['cpu']

# yf225 TODO: move to common utils?
TORCH_NN_MODULE_COMMON_TEST_HARNESS = """\n
#include <torch/script.h>

void write_ivalue_to_file(const torch::IValue& ivalue, const std::string& file_path) {
  auto bytes = torch::jit::pickle_save(ivalue);
  std::ofstream fout(file_path, std::ios::out | std::ios::binary);
  fout.write(bytes.data(), bytes.size());
  fout.close();
}
"""

TORCH_NN_MODULE_TEST_FORWARD_BACKWARD = Template("""\n
void ${module_variant_name}_test_forward_backward(const std::string& device) {
  pybind11::gil_scoped_release no_gil;

  torch::manual_seed(0);

  ${module_qualified_name} module${cpp_constructor_args};
  module->to(device);

  // Forward pass
  auto cpp_output = module(${cpp_input_args});

  // Save the output into a file to be compared in Python later
  write_ivalue_to_file(
    torch::IValue(cpp_output),
    "${cpp_output_tmp_folder}/${module_variant_name}_forward_output.pt");

  // Backward pass
  cpp_output.sum().backward();

  // Put all gradients into a c10::Dict, save it into a file to be compared in Python later
  // yf225 TODO: how are nested parameters named? Do they have proper prefix to avoid collision?
  c10::Dict<std::string, torch::Tensor> grad_dict;
  for (const auto& param : module->named_parameters()) {
    grad_dict.insert(param.key() + "_grad", param.value().grad());
  }
  write_ivalue_to_file(
    torch::IValue(grad_dict),
    "${cpp_output_tmp_folder}/${module_variant_name}_backward_grad_dict.pt");
}
""")

def _compile_cpp_code_inline(name, cpp_sources, functions):
    # Just-in-time compile the C++ test code
    cpp_module = torch.utils.cpp_extension.load_inline(
        name=name,
        cpp_sources=cpp_sources,
        functions=functions,
        verbose=False,
    )
    return cpp_module

def _test_torch_nn_module_variant(test_params):
    def set_cpp_tensors_all_requires_grad(cpp_inputs):
      return [x + ".requires_grad_()" for x in cpp_inputs]

    # yf225 TODO: move to common utils?
    # yf225 TODO: we should check in a copy of the generated source code, and then run consistency test (compare old vs. newly generated)
    def generate_test_cpp_sources(test_params, template):
      cpp_constructor_args = test_params.cpp_constructor_args
      if cpp_constructor_args != '':
        cpp_constructor_args = '({})'.format(cpp_constructor_args)
      test_cpp_sources = template.substitute(
        module_variant_name=test_params.module_variant_name,
        module_qualified_name='torch::nn::{}'.format(test_params.module_name),
        cpp_constructor_args=cpp_constructor_args,
        cpp_input_args=", ".join(set_cpp_tensors_all_requires_grad(test_params.cpp_input_args)),
        cpp_output_tmp_folder=test_params.cpp_output_tmp_folder,
      )
      print(test_cpp_sources) # yf225 TODO: turn this off when ready
      return test_cpp_sources

    def set_python_tensors_all_requires_grad(python_input):
      # Why is this function not working???
      if isinstance(python_input, torch.Tensor) and python_input.dtype != torch.long:
        return python_input.requires_grad_(True)
      else:
        return [set_python_tensors_all_requires_grad(tensor) for tensor in python_input]

    def test_forward_backward(test_params):
      # yf225 TODO: this function should handle all of forward-backward testing logic and do the result check
      torch.manual_seed(0)

      device = test_params.device
      python_constructor = test_params.test_instance.constructor
      python_constructor_args = test_params.test_instance.constructor_args

      module = python_constructor(*python_constructor_args).to(device)
      inputs = set_python_tensors_all_requires_grad(test_params.test_instance._get_input())
      python_output = module(inputs)

      python_output.sum().backward()
      # Put all gradients into a dict, to be compared later
      python_grad_dict = {}
      for name, param in module.named_parameters():
        python_grad_dict[name + "_grad"] = param.grad

      # yf225 TODO: maybe we can compile the cpp source code altogether first, and then do the result checking, to shorten the overall test time
      # bonus point: check in the cpp source code for comparison
      cpp_sources = TORCH_NN_MODULE_COMMON_TEST_HARNESS + \
        torch_nn_modules.module_metadata_map[test_params.module_name].cpp_sources + \
        generate_test_cpp_sources(
          test_params=test_params, template=TORCH_NN_MODULE_TEST_FORWARD_BACKWARD)

      cpp_module = _compile_cpp_code_inline(
        name=test_params.module_variant_name,
        cpp_sources=cpp_sources,
        functions=['{}_{}'.format(test_params.module_variant_name, 'test_forward_backward')])

      cpp_test_name = '{}_{}'.format(test_params.module_variant_name, 'test_forward_backward')
      cpp_test_fn = getattr(cpp_module, cpp_test_name)
      if not test_params.has_parity:
        with self.assertRaisesRegex(RuntimeError, "Parity test failed"):
          cpp_test_fn("cpu")
      else:
        cpp_test_fn("cpu")
        cpp_output = torch.load("{}/{}_forward_output.pt".format(test_params.cpp_output_tmp_folder, test_params.module_variant_name))
        cpp_grad_dict = torch.load("{}/{}_backward_grad_dict.pt".format(test_params.cpp_output_tmp_folder, test_params.module_variant_name))

        # yf225 TODO: use self.assertTrue ??? (We can pass in the unit_test_class object and call the method from there)
        assert torch.equal(python_output, cpp_output)
        assert len(python_grad_dict) == len(cpp_grad_dict)
        for key in python_grad_dict:
          assert key in cpp_grad_dict
          assert torch.equal(python_grad_dict[key], cpp_grad_dict[key])

      # Remove temporary folder that stores C++ outputs
      shutil.rmtree(test_params.cpp_output_tmp_folder)

    test_forward_backward(test_params)

# yf225 TODO: move to common utils?
def _compute_module_name(test_params_dict):
    fullname = test_params_dict.get('fullname', None)
    if fullname:
        # NOTE: This doesn't work for some of the `wrap_functional` module tests such as "interpolate_nearest_1d",
        # because in that case the module `interpolate` is not in `torch.nn` but rather in `torch.nn.functional`.
        # We will fix this when we have parity tests for `torch.nn.functional` modules.
        module_name = fullname.split('_')[0]
    else:
        module_name = test_params_dict.get('module_name')
    return module_name

# yf225 TODO: move to common utils?
def _process_test_params_for_module(test_params_dict, module_metadata, device, is_criterion):
    module_name = _compute_module_name(test_params_dict)
    test_params_dict['constructor'] = test_params_dict.get('constructor', getattr(torch.nn, module_name))
    if is_criterion:
        test = common_nn.CriterionTest(**test_params_dict)
    else:
        test = common_nn.ModuleTest(**test_params_dict)
    # yf225 TODO: can we remove the magic number `5` here?
    module_variant_name = test.get_name()[5:] + (('_' + device) if device != 'cpu' else '')    

    return TorchNNTestParams(
        module_name=module_name,
        module_variant_name=module_variant_name,
        test_instance=test,
        cpp_constructor_args=test_params_dict.get('cpp_constructor_args', ''),
        cpp_input_args=test_params_dict.get('cpp_input_args'),
        has_parity=test_params_dict.get('has_parity', True),
        device=device,
        cpp_output_tmp_folder=tempfile.mkdtemp(),
    )

# yf225 TODO: move to common utils?
def has_test(unit_test_class, test_name):
    return hasattr(unit_test_class, test_name)

# yf225 TODO: move to common utils?
def add_test(unit_test_class, test_name, test_fn):
    if has_test(unit_test_class, test_name):
        raise RuntimeError("Found two tests with the same name: " + test_name)
    setattr(unit_test_class, test_name, test_fn)

def add_torch_nn_module_impl_parity_tests(parity_table, unit_test_class, torch_nn_modules, module_tests, is_criterion):
  torch_nn_test_params_map = {}
  for test_params_dict in module_tests:
    # Skip all `torch.nn.functional` tests, since they are handled by another test suite.
    if 'FunctionalModule' in str(test_params_dict.get('constructor', '')):
      continue

    module_name = _compute_module_name(test_params_dict)

    assert hasattr(torch.nn, module_name), \
      "`torch.nn` doesn't have module `{}`. ".format(module_name) + \
      "If you are adding a new test, please set `fullname` using format `ModuleName_desc`, " + \
      "or set `module_name` using format `ModuleName`."

    module_full_name = 'torch::nn::' + module_name
    # If equivalent module in C++ frontend doesn't exist, we don't do the parity test.
    if module_full_name not in parity_table['torch::nn']:
      continue

    has_impl_parity, _ = parity_table['torch::nn'][module_full_name]

    def add_variant_test_for_module(module_name, test_params_dict, has_impl_parity, torch_nn_modules):
      module_metadata = torch_nn_modules.module_metadata_map[module_name]
      for device in devices:
        test_params = _process_test_params_for_module(
          test_params_dict=test_params_dict,
          module_metadata=module_metadata,
          device=device,
          is_criterion=is_criterion)
        test_name = 'test_torch_nn_{}'.format(test_params.module_variant_name)
        torch_nn_test_params_map[test_name] = test_params

        def test_fn(self):
          _test_torch_nn_module_variant(test_params=torch_nn_test_params_map[self._testMethodName])

        if device == 'cuda':
          test_fn = unittest.skipIf(not TEST_CUDA, "CUDA unavailable")(test_fn)

        # If `Implementation Parity` entry in parity table for this module is `No`,
        # we mark the test as expected failure.
        if not has_impl_parity:
          test_fn = unittest.expectedFailure(test_fn)

        add_test(unit_test_class, test_name, test_fn)

    add_variant_test_for_module(
      module_name=module_name,
      test_params_dict=test_params_dict,
      has_impl_parity=has_impl_parity,
      torch_nn_modules=torch_nn_modules)

def add_tests(unit_test_class, module_tests, criterion_tests, torch_nn_modules, parity_table):
  add_torch_nn_module_impl_parity_tests(
    parity_table=parity_table,
    unit_test_class=unit_test_class,
    torch_nn_modules=torch_nn_modules,
    module_tests=module_tests,
    is_criterion=False)

  add_torch_nn_module_impl_parity_tests(
    parity_table=parity_table,
    unit_test_class=unit_test_class,
    torch_nn_modules=torch_nn_modules,
    module_tests=criterion_tests,
    is_criterion=True)
