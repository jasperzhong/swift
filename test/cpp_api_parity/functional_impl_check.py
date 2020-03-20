import tempfile
import shutil
from string import Template
import unittest
from torch.testing._internal.common_cuda import TEST_CUDA

import torch
import torch.testing._internal.common_nn as common_nn
from cpp_api_parity.utils import TorchNNFunctionalTestParams, CppArg
from cpp_api_parity import torch_nn_functionals

# yf225 TODO: write better docs here

# Step 1: Translate ctor args from Python layer to C++ layer
# Step 2: Construct a C++ layer, run forward and backward on it, save all its params/buffers/gradients into a ScriptModule
# Step 3: Load that ScriptModule into Python, and compare output/params/buffers/gradients with Python layer (forward and backward)

# yf225 TODO: need to add `sample_functional` as sanity test

# NN tests use double as the default dtype
torch.set_default_dtype(torch.double)

# yf225 TODO: move to common utils?
devices = ['cpu', 'cuda']

# yf225 TODO: move to common utils?
TORCH_NN_COMMON_TEST_HARNESS = """
#include <torch/script.h>

void write_ivalue_to_file(const torch::IValue& ivalue, const std::string& file_path) {
  auto bytes = torch::jit::pickle_save(ivalue);
  std::ofstream fout(file_path, std::ios::out | std::ios::binary);
  fout.write(bytes.data(), bytes.size());
  fout.close();
}

c10::Dict<std::string, torch::Tensor> load_dict_from_file(const std::string& file_path) {
  c10::Dict<std::string, torch::Tensor> arg_dict;
  auto arg_dict_module = torch::jit::load(file_path);
  for (const auto& p : arg_dict_module.named_buffers(/*recurse=*/false)) {
    arg_dict.insert(p.name, p.value);
  }
  return arg_dict;
}
"""

'''
Expected substitutions:

${functional_variant_name}
${cpp_tmp_folder}
${cpp_args_construction_stmts}
${cpp_function_call}
'''
TORCH_NN_FUNCTIONAL_TEST_FORWARD = Template("""
void ${functional_variant_name}_test_forward() {
  pybind11::gil_scoped_release no_gil;

  namespace F = torch::nn::functional;

  // Declare arguments
  auto arg_dict = load_dict_from_file("${cpp_tmp_folder}/${functional_variant_name}_arg_dict.pt");
  ${cpp_args_construction_stmts};

  // Some functionals (such as `F::rrelu`) create random tensors in their call path.
  // To make sure the random tensors created are the same in Python/C++, we need
  // to set the RNG seed manually.
  torch::manual_seed(0);

  // Run function with arguments
  auto cpp_output = ${cpp_function_call};

  // Save the output into a file to be compared in Python later
  write_ivalue_to_file(
    torch::IValue(cpp_output),
    "${cpp_tmp_folder}/${functional_variant_name}_forward_output.pt");
}
""")

# yf225 TODO: move to common utils?
def compile_cpp_code_inline(name, cpp_sources, functions):
  cpp_module = torch.utils.cpp_extension.load_inline(
    name=name,
    cpp_sources=cpp_sources,
    functions=functions,
    verbose=False,
  )
  return cpp_module

# yf225 TODO: move to common utils
def convert_to_list(python_input):
  if isinstance(python_input, torch.Tensor):
    return [python_input]
  else:
    return [tensor for tensor in python_input]

# yf225 TODO: move to common utils
def set_python_tensors_requires_grad(python_tensors):
  return [tensor.requires_grad_(True) if tensor.dtype != torch.long else tensor for tensor in python_tensors]

# yf225 TODO: move to common utils
def move_python_tensors_to_device(python_tensors, device):
  return [tensor.to(device) for tensor in python_tensors]

def run_forward(unit_test_class, test_params):
  device = test_params.device
  
  inputs = set_python_tensors_requires_grad([arg_value for _, arg_value in test_params.arg_dict['input']])
  inputs = inputs + [arg_value for _, arg_value in test_params.arg_dict['target']]
  inputs = inputs + [arg_value for _, arg_value in test_params.arg_dict['extra_args']]
  inputs = move_python_tensors_to_device(inputs, device)

  # Some functionals (such as `F.rrelu`) create random tensors in their call path.
  # To make sure the random tensors created are the same in Python/C++, we need
  # to set the RNG seed manually.
  torch.manual_seed(0)
  python_output = test_params.test_instance.constructor()(*inputs)

  return python_output

def test_forward(unit_test_class, test_params):
  functional_variant_name = test_params.functional_variant_name

  # Run forward on Python functional
  python_output = run_forward(unit_test_class, test_params)

  # Save Python arguments to be used from C++ function
  arg_dict_flat = {
    arg_name: arg_value \
      for arg_name, arg_value in \
        test_params.arg_dict['input'] + \
        test_params.arg_dict['target'] + \
        test_params.arg_dict['extra_args'] + \
        test_params.arg_dict['other']
  }
  arg_dict_module = torch.nn.Module()
  for arg_name, arg_value in arg_dict_flat.items():
    assert isinstance(arg_value, torch.Tensor)
    arg_dict_module.register_buffer(arg_name, arg_value)
  torch.jit.script(arg_dict_module).save("{}/{}_arg_dict.pt".format(test_params.cpp_tmp_folder, functional_variant_name))

  cpp_test_name = '{}_{}'.format(test_params.functional_variant_name, 'test_forward')
  cpp_test_fn = getattr(unit_test_class.functional_impl_check_cpp_module, cpp_test_name)

  def run_cpp_test_fn_and_check_output():
    cpp_test_fn()
    cpp_output = torch.load("{}/{}_forward_output.pt".format(test_params.cpp_tmp_folder, functional_variant_name))

    def generate_error_msg(name, cpp_value, python_value):
      return "Parity test failed: {} in C++ has value: {}, which does not match the corresponding value in Python: {}".format(
        name, cpp_value, python_value)

    # Check that forward outputs are equal
    unit_test_class.assertTrue(
      torch.allclose(python_output, cpp_output),
      generate_error_msg("forward output", cpp_output, python_output))

  if not test_params.has_parity:
    with unit_test_class.assertRaisesRegex(AssertionError, "Parity test failed"):
      run_cpp_test_fn_and_check_output()
  else:
    run_cpp_test_fn_and_check_output()

  # Remove temporary folder that stores C++ outputs
  shutil.rmtree(test_params.cpp_tmp_folder)

def test_torch_nn_functional_variant(unit_test_class, test_params):
  test_forward(unit_test_class, test_params)

# yf225 TODO: move to common utils?
def compute_functional_name(test_params_dict):
  if 'cpp_options_args' in test_params_dict:
    return test_params_dict['constructor']().fn.__name__
  elif 'cpp_function_call' in test_params_dict:
    # Expected format: `F::functional_name(...)`
    return test_params_dict['cpp_function_call'].replace('F::', '').split('(')[0]
  else:
    raise RuntimeError(
      "`cpp_options_args` or `cpp_function_call` entry must be present in test params dict: {}".format(
        test_params_dict))

def compute_cpp_function_call(test_params_dict, arg_dict, functional_name):
  if 'cpp_function_call' in test_params_dict:
    return test_params_dict['cpp_function_call']
  elif 'cpp_options_args' in test_params_dict:
    cpp_forward_args_symbols = [arg_name for arg_name, _ in arg_dict['input'] + arg_dict['target'] + arg_dict['extra_args']]
    return 'F::{}({}, {})'.format(functional_name, ", ".join(cpp_forward_args_symbols), test_params_dict['cpp_options_args'])
  else:
    raise RuntimeError(
      "`cpp_options_args` or `cpp_function_call` entry must be present in test params dict: {}".format(
        test_params_dict))


# yf225 TODO: move to common utils?
def process_test_params_for_functional(test_params_dict, functional_metadata, device, test_instance_class):
  test = test_instance_class(**test_params_dict)
  # yf225 TODO: can we remove the magic number `5` here?
  functional_name = compute_functional_name(test_params_dict)
  functional_variant_name = test.get_name()[5:] + (('_' + device) if device != 'cpu' else '')

  # yf225 TODO: put cpp_arg_symbol_map processing into a common util function!
  arg_dict = {
    'input': [],
    'target': [],
    'extra_args': [],
    'other': [],
  }

  def put_args_into_arg_dict(arg_type, arg_type_prefix, args):
    for i, arg in enumerate(args):
      arg_dict[arg_type].append(CppArg(name=arg_type_prefix+str(i), value=arg))

  put_args_into_arg_dict('input', 'i', convert_to_list(test._get_input()))
  if is_criterion_test(test):
    put_args_into_arg_dict('target', 't', convert_to_list(test._get_target()))
  if test.extra_args:
    put_args_into_arg_dict('extra_args', 'e', convert_to_list(test.extra_args))

  cpp_arg_symbol_map = test_params_dict.get('cpp_arg_symbol_map', {})
  for arg_name, arg_value in cpp_arg_symbol_map.items():
    if isinstance(arg_value, str):
      if arg_value == 'input':
        arg_dict['other'].append(CppArg(name=arg_name, value=test._get_input()))
      else:
        raise RuntimeError("`{}` has unsupported string value: {}".format(arg_name, arg_value))
    elif isinstance(arg_value, torch.Tensor):
      arg_dict['other'].append(CppArg(name=arg_name, value=arg_value))
    else:
      raise RuntimeError("`{}` has unsupported value: {}".format(arg_name, arg_value))

  return TorchNNFunctionalTestParams(
    functional_name=functional_name,
    functional_variant_name=functional_variant_name,
    test_instance=test,
    cpp_function_call=compute_cpp_function_call(test_params_dict, arg_dict, functional_name),
    arg_dict=arg_dict,
    has_parity=test_params_dict.get('has_parity', True),
    device=device,
    cpp_tmp_folder=tempfile.mkdtemp(),
  )

# yf225 TODO: move to common utils?
def has_test(unit_test_class, test_name):
  return hasattr(unit_test_class, test_name)

# yf225 TODO: move to common utils?
def add_test(unit_test_class, test_name, test_fn):
  if has_test(unit_test_class, test_name):
    raise RuntimeError("Found two tests with the same name: " + test_name)
  setattr(unit_test_class, test_name, test_fn)

# yf225 TODO: move to common utils?
def set_cpp_tensors_requires_grad(cpp_tensor_stmts, cpp_tensors):
  assert len(cpp_tensor_stmts) == len(cpp_tensors)
  return ['{}.requires_grad_(true)'.format(tensor_stmt) if tensor.dtype != torch.long else tensor_stmt \
    for tensor_stmt, (_, tensor) in zip(cpp_tensor_stmts, cpp_tensors)]

# yf225 TODO: move to common utils
def move_cpp_tensors_to_device(cpp_tensor_stmts, device):
  return ['{}.to("{}")'.format(tensor_stmt, device) for tensor_stmt in cpp_tensor_stmts]

def is_criterion_test(test_instance):
  return isinstance(test_instance, common_nn.CriterionTest) or \
    isinstance(test_instance, common_nn.NewCriterionTest)


torch_nn_test_params_map = {}

def add_torch_nn_functional_impl_parity_tests(parity_table, unit_test_class, test_params_dicts, test_instance_class):
  for test_params_dict in test_params_dicts:
    # Skip all `torch.nn` module tests, since they are handled by another test suite.
    if not 'FunctionalModule' in str(test_params_dict.get('constructor', '')):
      continue

    functional_name = compute_functional_name(test_params_dict)

    assert hasattr(torch.nn.functional, functional_name), \
      "`torch.nn.functional` doesn't have function `{}` (discovered while processing {})".format(
        functional_name, test_params_dict)

    functional_full_name = 'F::' + functional_name

    assert functional_full_name in parity_table['torch::nn::functional'], \
      "Please add `{}` entry to `torch::nn::functional` section of `test/cpp_api_parity/parity-tracker.md` (discovered while processing {})".format(
        functional_full_name, test_params_dict)

    has_impl_parity, _ = parity_table['torch::nn::functional'][functional_full_name]

    functional_metadata = torch_nn_functionals.functional_metadata_map.get(functional_name, torch_nn_functionals.TorchNNFunctionalMetadata())
    for device in devices:
      test_params = process_test_params_for_functional(
        test_params_dict=test_params_dict,
        functional_metadata=functional_metadata,
        device=device,
        test_instance_class=test_instance_class,
      )
      test_name = 'test_torch_nn_functional_{}'.format(test_params.functional_variant_name)
      torch_nn_test_params_map[test_name] = test_params

      def test_fn(self):
        test_torch_nn_functional_variant(unit_test_class=self, test_params=torch_nn_test_params_map[self._testMethodName])

      if device == 'cuda':
        test_fn = unittest.skipIf(not TEST_CUDA, "CUDA unavailable")(test_fn)

      # If `Implementation Parity` entry in parity table for this functional is `No`,
      # we mark the test as expected failure.
      if not has_impl_parity:
        test_fn = unittest.expectedFailure(test_fn)

      add_test(unit_test_class, test_name, test_fn)


def add_tests(unit_test_class, test_params_dicts, test_instance_class, parity_table):
  add_torch_nn_functional_impl_parity_tests(
    parity_table=parity_table,
    unit_test_class=unit_test_class,
    test_params_dicts=test_params_dicts,
    test_instance_class=test_instance_class)

# yf225 TODO: move to common utils?
# yf225 TODO: we should check in a copy of the generated source code, and then run consistency test (compare old vs. newly generated)
def generate_test_cpp_sources(test_params, template):
  device = test_params.device

  # yf225 TODO: move to common utils!
  def add_cpp_forward_args(args):
    args_stmts = []
    for arg_name, _ in args:
      args_stmts.append('auto {} = arg_dict.at("{}")'.format(arg_name, arg_name))
    return args_stmts

  cpp_forward_input_args_stmts = move_cpp_tensors_to_device(set_cpp_tensors_requires_grad(add_cpp_forward_args(test_params.arg_dict['input']), test_params.arg_dict['input']), device)
  cpp_forward_target_args_stmts = move_cpp_tensors_to_device(add_cpp_forward_args(test_params.arg_dict['target']), device)
  cpp_forward_extra_args_stmts = move_cpp_tensors_to_device(add_cpp_forward_args(test_params.arg_dict['extra_args']), device)

  # Build the list of other arguments needed
  cpp_other_args_stmts = []
  for arg_name, _ in test_params.arg_dict['other']:
    cpp_other_args_stmts.append('auto {} = arg_dict.at("{}")'.format(arg_name, arg_name))
  cpp_other_args_stmts = move_cpp_tensors_to_device(cpp_other_args_stmts, device)
  
  cpp_args_construction_stmts = cpp_forward_input_args_stmts + cpp_forward_target_args_stmts + cpp_forward_extra_args_stmts + cpp_other_args_stmts

  test_cpp_sources = template.substitute(
    functional_variant_name=test_params.functional_variant_name,
    cpp_args_construction_stmts=";\n  ".join(cpp_args_construction_stmts),
    cpp_function_call=test_params.cpp_function_call,
    cpp_tmp_folder=test_params.cpp_tmp_folder,
  )
  return test_cpp_sources

def build_cpp_tests(unit_test_class):
  # Put all cpp source code into one file and compile together, in order to speed up the build
  # yf225 TODO bonus point: check in the cpp source code for comparison
  if len(torch_nn_test_params_map) > 0:
    cpp_sources = TORCH_NN_COMMON_TEST_HARNESS
    functions = []
    functionals_added_metadata_cpp_sources = set()
    for test_name, test_params in torch_nn_test_params_map.items():
      if not test_params.functional_name in functionals_added_metadata_cpp_sources:
        cpp_sources += torch_nn_functionals.functional_metadata_map.get(test_params.functional_name, torch_nn_functionals.TorchNNFunctionalMetadata()).cpp_sources
        functionals_added_metadata_cpp_sources.add(test_params.functional_name)
      cpp_sources += generate_test_cpp_sources(test_params=test_params, template=TORCH_NN_FUNCTIONAL_TEST_FORWARD)
      functions.append('{}_{}'.format(test_params.functional_variant_name, 'test_forward'))
    print(cpp_sources)  # yf225 TODO: remove this when ready

    cpp_module = compile_cpp_code_inline(
      name='functional_impl_check',
      cpp_sources=cpp_sources,
      functions=functions)
    unit_test_class.functional_impl_check_cpp_module = cpp_module