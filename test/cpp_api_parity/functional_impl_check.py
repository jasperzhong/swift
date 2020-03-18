import tempfile
import shutil
from string import Template
import unittest
from torch.testing._internal.common_cuda import TEST_CUDA

import torch
import torch.testing._internal.common_nn as common_nn
from cpp_api_parity.utils import TorchNNFunctionalTestParams
from cpp_api_parity import torch_nn_functionals

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
"""

TORCH_NN_FUNCTIONAL_TEST = Template("""
void ${functional_variant_name}_test() {
  pybind11::gil_scoped_release no_gil;

  namespace F = torch::nn::functional;

  // NOTE: Because different RNG state would lead to different output,
  // it is crucial for this function to execute the same statements
  // in the exact same order as the Python equivalent, otherwise their
  // outputs would not be the same.
  torch::manual_seed(0);

  ${cpp_input_args_construction_stmts}
  auto cpp_output = F::${functional_name}(${cpp_input_args_symbols}${cpp_options_arg});

  // Save the output into a file to be compared in Python later
  write_ivalue_to_file(
    torch::IValue(cpp_output),
    "${cpp_output_tmp_folder}/${functional_variant_name}_forward_output.pt");
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

def _test_torch_nn_functional_variant(unit_test_class, test_params):
  # yf225 TODO: move to common utils?
  def convert_to_list(python_input):
    if isinstance(python_input, torch.Tensor):
      return [python_input]
    else:
      return [tensor for tensor in python_input]

  # yf225 TODO: move to common utils
  def move_python_tensors_to_device(python_tensors, device):
    return [tensor.to(device) for tensor in python_tensors]

  def do_test(unit_test_class, test_params):
    # NOTE: Because different RNG state would lead to different output,
    # it is crucial for this function to execute the same statements
    # in the exact same order as the C++ equivalent, otherwise their
    # outputs would not be the same.
    torch.manual_seed(0)

    device = test_params.device
    inputs = convert_to_list(test_params.test_instance._get_input()) # yf225 TODO: convert inputs to CUDA device when needed!
    if is_criterion_test(test_params.test_instance):
      inputs = inputs + convert_to_list(test_params.test_instance._get_target())
      inputs = inputs + convert_to_list(test_params.test_instance.extra_args)
    inputs = move_python_tensors_to_device(inputs, device)
    python_output = test_params.test_instance.constructor()(*inputs)

    cpp_test_name = '{}_{}'.format(test_params.functional_variant_name, 'test')
    cpp_test_fn = getattr(unit_test_class.functional_impl_check_cpp_module, cpp_test_name)

    def run_cpp_test_fn_and_check_output():
      cpp_test_fn()
      cpp_output = torch.load("{}/{}_forward_output.pt".format(test_params.cpp_output_tmp_folder, test_params.functional_variant_name))

      def generate_error_msg(name, cpp_value, python_value):
        return "Parity test failed: {} in C++ has value: {}, which does not match the corresponding value in Python: {}".format(
          name, cpp_value, python_value)

      # Check that forward outputs are equal
      unit_test_class.assertTrue(
        torch.equal(python_output, cpp_output),
        generate_error_msg("forward output", cpp_output, python_output))

    if not test_params.has_parity:
      with unit_test_class.assertRaisesRegex(AssertionError, "Parity test failed"):
        run_cpp_test_fn_and_check_output()
    else:
      run_cpp_test_fn_and_check_output()

    # Remove temporary folder that stores C++ outputs
    shutil.rmtree(test_params.cpp_output_tmp_folder)

  do_test(unit_test_class, test_params)

# yf225 TODO: move to common utils?
# yf225 TODO: need to add `sample_functional` function
def _process_test_params_for_functional(test_params_dict, functional_metadata, device, test_instance_class):
  functional_name = test_params_dict['functional_name']
  test = test_instance_class(**test_params_dict)
  # yf225 TODO: can we remove the magic number `5` here?
  functional_variant_name = test.get_name()[5:] + (('_' + device) if device != 'cpu' else '')    
  assert "cpp_input_args" in test_params_dict, \
    "`cpp_input_args` entry must be present in test params dict for {}".format(functional_variant_name)

  return TorchNNFunctionalTestParams(
    functional_name=functional_name,
    functional_variant_name=functional_variant_name,
    test_instance=test,
    cpp_options_arg=test_params_dict.get('cpp_options_arg', ''),
    cpp_input_args=test_params_dict['cpp_input_args'],
    cpp_target_args=test_params_dict.get('cpp_target_args', None),
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

# yf225 TODO: move to common utils?
def is_criterion_test(test_instance):
  return isinstance(test_instance, common_nn.CriterionTest) or \
    isinstance(test_instance, common_nn.NewCriterionTest)

# yf225 TODO: move to common utils
def move_cpp_tensors_to_device(cpp_tensors, device):
  return ['{}.to(std::string("{}"))'.format(tensor, device) for tensor in cpp_tensors]

# yf225 TODO: move to common utils?
# yf225 TODO: we should check in a copy of the generated source code, and then run consistency test (compare old vs. newly generated)
def generate_test_cpp_sources(test_params, template):
  cpp_options_arg = test_params.cpp_options_arg
  if cpp_options_arg != '':
    cpp_options_arg = ', {}'.format(cpp_options_arg)

  cpp_input_args = test_params.cpp_input_args
  if is_criterion_test(test_params.test_instance):
    assert test_params.cpp_target_args is not None, \
      "`cpp_target_args` entry must be present in test params dict for {}".format(test_params.functional_variant_name)
    cpp_input_args = cpp_input_args + test_params.cpp_target_args
    if test_params.cpp_extra_args:
      cpp_input_args = cpp_input_args + test_params.cpp_extra_args
  cpp_input_args = move_cpp_tensors_to_device(cpp_input_args, test_params.device)

  cpp_input_args_construction_stmts = []
  cpp_input_args_symbols = []
  for i, input_arg in enumerate(cpp_input_args):
    cpp_input_args_construction_stmts.append("auto i{} = {};".format(i, input_arg))
    cpp_input_args_symbols.append("i{}".format(i))

  test_cpp_sources = template.substitute(
    functional_variant_name=test_params.functional_variant_name,
    functional_name=test_params.functional_name,
    cpp_input_args_construction_stmts="\n  ".join(cpp_input_args_construction_stmts),
    cpp_input_args_symbols=", ".join(cpp_input_args_symbols),
    cpp_options_arg=cpp_options_arg,
    cpp_output_tmp_folder=test_params.cpp_output_tmp_folder,
  )
  return test_cpp_sources

torch_nn_test_params_map = {}

def add_torch_nn_functional_impl_parity_tests(parity_table, unit_test_class, test_params_dicts, test_instance_class):
  for test_params_dict in test_params_dicts:
    # Skip all `torch.nn` module tests, since they are handled by another test suite.
    if not 'FunctionalModule' in str(test_params_dict.get('constructor', '')):
      continue

    assert 'functional_name' in test_params_dict, \
      "`functional_name` entry must be present in test params dict for {}".format(test_params_dict['fullname'])
    functional_name = test_params_dict['functional_name']

    assert hasattr(torch.nn.functional, functional_name), \
      "`torch.nn.functional` doesn't have function `{}`. ".format(functional_name) + \
      "If you are adding a new test, please set `functional_name` field in the test params dict of {}.".format(test_params_dict['fullname'])

    functional_full_name = 'F::' + functional_name
    # If equivalent functional in C++ frontend doesn't exist, we don't do the parity test.
    if functional_full_name not in parity_table['torch::nn::functional']:
      continue

    has_impl_parity, _ = parity_table['torch::nn::functional'][functional_full_name]
    # yf225 TODO: here we should just skip if there is no parity!!!! don't mark it as expected failure!!!

    functional_metadata = torch_nn_functionals.functional_metadata_map.get(functional_name, torch_nn_functionals.TorchNNFunctionalMetadata())
    for device in devices:
      test_params = _process_test_params_for_functional(
        test_params_dict=test_params_dict,
        functional_metadata=functional_metadata,
        device=device,
        test_instance_class=test_instance_class,
      )
      test_name = 'test_torch_nn_functional_{}'.format(test_params.functional_variant_name)
      torch_nn_test_params_map[test_name] = test_params

      def test_fn(self):
        _test_torch_nn_functional_variant(unit_test_class=self, test_params=torch_nn_test_params_map[self._testMethodName])

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
      cpp_sources += generate_test_cpp_sources(test_params=test_params, template=TORCH_NN_FUNCTIONAL_TEST)
      functions.append('{}_{}'.format(test_params.functional_variant_name, 'test'))
    print(cpp_sources)  # yf225 TODO: remove this when ready

    cpp_module = _compile_cpp_code_inline(
      name='functional_impl_check',
      cpp_sources=cpp_sources,
      functions=functions)
    unit_test_class.functional_impl_check_cpp_module = cpp_module
