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
devices = ['cpu', 'cuda']

# yf225 TODO: move to common utils?
TORCH_NN_MODULE_COMMON_TEST_HARNESS = """
#include <torch/script.h>

void write_ivalue_to_file(const torch::IValue& ivalue, const std::string& file_path) {
  auto bytes = torch::jit::pickle_save(ivalue);
  std::ofstream fout(file_path, std::ios::out | std::ios::binary);
  fout.write(bytes.data(), bytes.size());
  fout.close();
}

// Generates rand tensor with non-equal values. This ensures that duplicate
// values won't be causing test failure for modules like MaxPooling.
// size should be small, otherwise randperm fails / long overflows.
torch::Tensor _rand_tensor_non_equal(torch::IntArrayRef size) {
  int64_t total = 1;
  for (int64_t elem : size) {
    total *= elem;
  }
  return torch::randperm(total).view(size).to(torch::kDouble);
}
"""

TORCH_NN_MODULE_TEST_FORWARD_BACKWARD = Template("""
void ${module_variant_name}_test_forward_backward(const std::string& device) {
  pybind11::gil_scoped_release no_gil;

  // NOTE: Because different RNG state would lead to different output,
  // it is crucial for this function to execute the same statements
  // in the exact same order as the Python equivalent, otherwise their
  // outputs would not be the same.
  torch::manual_seed(0);

  ${module_qualified_name} module${cpp_constructor_args};
  module->to(device);

  // Forward pass
  ${cpp_input_args_construction_stmts}
  auto cpp_output = module(${cpp_input_args_symbols});

  // Save the output into a file to be compared in Python later
  write_ivalue_to_file(
    torch::IValue(cpp_output),
    "${cpp_output_tmp_folder}/${module_variant_name}_forward_output.pt");

  // Backward pass
  cpp_output.sum().backward();

  // Put all gradients into a c10::Dict, save it into a file to be compared in Python later
  c10::Dict<std::string, torch::Tensor> grad_dict;
  for (const auto& param : module->named_parameters()) {
    torch::Tensor grad = param.value().grad();
    if (grad.is_sparse()) {
      grad = grad.to_dense();
    }
    grad_dict.insert(param.key() + "_grad", grad);
  }

  write_ivalue_to_file(
    torch::IValue(grad_dict),
    "${cpp_output_tmp_folder}/${module_variant_name}_backward_grad_dict.pt");
}
""")

# yf225 TODO: move to common utils?
def _compile_cpp_code_inline(name, cpp_sources, functions):
  cpp_module = torch.utils.cpp_extension.load_inline(
    name=name,
    cpp_sources=cpp_sources,
    functions=functions,
    verbose=False,
  )
  return cpp_module

def _test_torch_nn_module_variant(unit_test_class, test_params):
  def convert_to_list(python_input):
    if isinstance(python_input, torch.Tensor):
      return [python_input]
    else:
      return [tensor for tensor in python_input]

  def set_python_tensors_all_requires_grad(python_tensors):
    # yf225 TODO: we might also need to cast inputs to CUDA for CUDA tests
    return [tensor.requires_grad_(True) if tensor.dtype != torch.long else tensor for tensor in python_tensors]

  def test_forward_backward(unit_test_class, test_params):
    # NOTE: Because different RNG state would lead to different output,
    # it is crucial for this function to execute the same statements
    # in the exact same order as the C++ equivalent, otherwise their
    # outputs would not be the same.
    torch.manual_seed(0)

    device = test_params.device
    module = test_params.test_instance.constructor(*test_params.test_instance.constructor_args).to(device)
    inputs = set_python_tensors_all_requires_grad(convert_to_list(test_params.test_instance._get_input()))
    if is_criterion_test(test_params.test_instance):
      inputs += [test_params.test_instance._get_target()]
      inputs += convert_to_list(test_params.test_instance.extra_args)
    python_output = module(*inputs)

    python_output.sum().backward()
    # Put all gradients into a dict, to be compared later
    python_grad_dict = {}
    for name, param in module.named_parameters():
      grad = param.grad;
      if grad.is_sparse:
        grad = grad.to_dense()
      python_grad_dict[name + "_grad"] = grad

    cpp_test_name = '{}_{}'.format(test_params.module_variant_name, 'test_forward_backward')
    cpp_test_fn = getattr(unit_test_class.cpp_module, cpp_test_name)

    def run_cpp_test_fn_and_check_output():
      cpp_test_fn(device)
      cpp_output = torch.load("{}/{}_forward_output.pt".format(test_params.cpp_output_tmp_folder, test_params.module_variant_name))
      cpp_grad_dict = torch.load("{}/{}_backward_grad_dict.pt".format(test_params.cpp_output_tmp_folder, test_params.module_variant_name))

      def generate_error_msg(name, cpp_value, python_value):
        return "Parity test failed: {} in C++ has value: {}, which does not match the corresponding value in Python: {}".format(
          name, cpp_value, python_value)

      # Check that forward outputs are equal
      unit_test_class.assertTrue(
        torch.equal(python_output, cpp_output),
        generate_error_msg("forward output", cpp_output, python_output))

      # Check that module parameter gradients are equal after backward pass
      unit_test_class.assertEqual(
        len(python_grad_dict), len(cpp_grad_dict),
        generate_error_msg("# of parameters", len(cpp_grad_dict), len(python_grad_dict)))
      for key in python_grad_dict:
        unit_test_class.assertTrue(
          key in cpp_grad_dict,
          generate_error_msg("\"Does module have a parameter named `{}`?\"".format(key[:-5]), False, True))
        unit_test_class.assertTrue(
          torch.equal(python_grad_dict[key], cpp_grad_dict[key]),
          generate_error_msg("gradient of `{}`".format(key[:-5]), cpp_grad_dict[key], python_grad_dict[key]))

    if not test_params.has_parity:
      with unit_test_class.assertRaisesRegex(AssertionError, "Parity test failed"):
        run_cpp_test_fn_and_check_output()
    else:
      run_cpp_test_fn_and_check_output()

    # Remove temporary folder that stores C++ outputs
    shutil.rmtree(test_params.cpp_output_tmp_folder)

  test_forward_backward(unit_test_class, test_params)

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
def _process_test_params_for_module(test_params_dict, module_metadata, device, test_instance_class):
  module_name = _compute_module_name(test_params_dict)
  test_params_dict['constructor'] = test_params_dict.get('constructor', getattr(torch.nn, module_name))
  test = test_instance_class(**test_params_dict)
  # yf225 TODO: can we remove the magic number `5` here?
  module_variant_name = test.get_name()[5:] + (('_' + device) if device != 'cpu' else '')    
  assert "cpp_input_args" in test_params_dict, \
    "`cpp_input_args` entry must be present in test params dict for {}".format(module_variant_name)

  return TorchNNTestParams(
    module_name=module_name,
    module_variant_name=module_variant_name,
    test_instance=test,
    cpp_constructor_args=test_params_dict.get('cpp_constructor_args', ''),
    cpp_input_args=test_params_dict['cpp_input_args'],
    cpp_target_args=test_params_dict.get('cpp_target_args', None),
    cpp_input_args_requires_grad=test_params_dict.get('cpp_input_args_requires_grad', True),
    cpp_extra_args=test_params_dict.get('cpp_extra_args', None),
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

def set_cpp_tensors_all_requires_grad(cpp_input_args):
  # yf225 TODO: we need a flag to decide whether to set requires grad (e.g. it doesn't work for long tensors)
  return ["{}.requires_grad_(true)".format(x) for x in cpp_input_args]

def is_criterion_test(test_instance):
  return isinstance(test_instance, common_nn.CriterionTest) or \
    isinstance(test_instance, common_nn.NewCriterionTest)

# yf225 TODO: move to common utils?
# yf225 TODO: we should check in a copy of the generated source code, and then run consistency test (compare old vs. newly generated)
def generate_test_cpp_sources(test_params, template):
  cpp_constructor_args = test_params.cpp_constructor_args
  if cpp_constructor_args != '':
    cpp_constructor_args = '({})'.format(cpp_constructor_args)

  cpp_input_args = test_params.cpp_input_args
  if test_params.cpp_input_args_requires_grad:
    cpp_input_args = set_cpp_tensors_all_requires_grad(cpp_input_args)
  if is_criterion_test(test_params.test_instance):
    assert test_params.cpp_target_args is not None, \
      "`cpp_target_args` entry must be present in test params dict for {}".format(test_params.module_variant_name)
    cpp_input_args += test_params.cpp_target_args
    if test_params.cpp_extra_args:
      cpp_input_args += test_params.cpp_extra_args

  cpp_input_args_construction_stmts = []
  cpp_input_args_symbols = []
  for i, input_arg in enumerate(cpp_input_args):
    cpp_input_args_construction_stmts.append("auto i{} = {};".format(i, input_arg))
    cpp_input_args_symbols.append("i{}".format(i))

  test_cpp_sources = template.substitute(
    module_variant_name=test_params.module_variant_name,
    module_qualified_name='torch::nn::{}'.format(test_params.module_name),
    cpp_constructor_args=cpp_constructor_args,
    cpp_input_args_construction_stmts="\n  ".join(cpp_input_args_construction_stmts),
    cpp_input_args_symbols=", ".join(cpp_input_args_symbols),
    cpp_output_tmp_folder=test_params.cpp_output_tmp_folder,
  )
  return test_cpp_sources

torch_nn_test_params_map = {}

def add_torch_nn_module_impl_parity_tests(parity_table, unit_test_class, torch_nn_modules, test_params_dicts, test_instance_class):
  for test_params_dict in test_params_dicts:
    print()
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

    module_metadata = torch_nn_modules.module_metadata_map[module_name]
    for device in devices:
      test_params = _process_test_params_for_module(
        test_params_dict=test_params_dict,
        module_metadata=module_metadata,
        device=device,
        test_instance_class=test_instance_class,
      )
      test_name = 'test_torch_nn_{}'.format(test_params.module_variant_name)
      torch_nn_test_params_map[test_name] = test_params

      def test_fn(self):
        _test_torch_nn_module_variant(unit_test_class=self, test_params=torch_nn_test_params_map[self._testMethodName])

      if device == 'cuda':
        test_fn = unittest.skipIf(not TEST_CUDA, "CUDA unavailable")(test_fn)

      # If `Implementation Parity` entry in parity table for this module is `No`,
      # we mark the test as expected failure.
      if not has_impl_parity:
        test_fn = unittest.expectedFailure(test_fn)

      add_test(unit_test_class, test_name, test_fn)


def add_tests(unit_test_class, test_params_dicts, test_instance_class, torch_nn_modules, parity_table):
  add_torch_nn_module_impl_parity_tests(
    parity_table=parity_table,
    unit_test_class=unit_test_class,
    torch_nn_modules=torch_nn_modules,
    test_params_dicts=test_params_dicts,
    test_instance_class=test_instance_class)

def build_cpp_tests(unit_test_class):
  # Put all cpp source code into one file and compile together, in order to speed up the build
  # yf225 TODO bonus point: check in the cpp source code for comparison
  if len(torch_nn_test_params_map) > 0:
    cpp_sources = TORCH_NN_MODULE_COMMON_TEST_HARNESS
    functions = []
    modules_added_metadata_cpp_sources = set()
    for test_name, test_params in torch_nn_test_params_map.items():
      if not test_params.module_name in modules_added_metadata_cpp_sources:
        cpp_sources += torch_nn_modules.module_metadata_map[test_params.module_name].cpp_sources
        modules_added_metadata_cpp_sources.add(test_params.module_name)
      cpp_sources += generate_test_cpp_sources(test_params=test_params, template=TORCH_NN_MODULE_TEST_FORWARD_BACKWARD)
      functions.append('{}_{}'.format(test_params.module_variant_name, 'test_forward_backward'))
    print(cpp_sources)  # yf225 TODO: remove this when ready

    cpp_module = _compile_cpp_code_inline(
      name='module_impl_check',
      cpp_sources=cpp_sources,
      functions=functions)
    unit_test_class.cpp_module = cpp_module
