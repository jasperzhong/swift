import torch

TORCH_NN_MODULE_COMMON_TEST_HARNESS = """\n
#include <torch/script.h>

const char * const parity_test_error_msg_prefix = "Parity test failed: ";

#define GENERATE_PARITY_TEST_ERROR_MSG(name, cpp_value, python_value) \
  parity_test_error_msg_prefix, \
  name, " in C++ has value: ", cpp_value, ", which does not match the corresponding value in Python: ", python_value \

bool check_tensor_equality(const torch::Tensor& tensor1, const torch::Tensor& tensor2) {
  return tensor1.sizes().vec() == tensor2.sizes().vec() && \
    tensor1.device() == tensor2.device() && \
    tensor1.dtype() == tensor2.dtype() && \
    tensor1.allclose(tensor2);
}

bool check_ivalue_equality(const c10::IValue& ivalue_python, const c10::IValue& ivalue_cpp) {
  // For Python modules, we allow the use of `int` to represent attributes that
  // are multidimensional but have the same value in all dimensions. The corresponding
  // data type for C++ modules is `ExpandingArray` (which is converted to `IntList` by the
  // `IValue` constructor), and here we check that all elements in the `ExpandingArray`
  // are equal to the Python `int` attribute.
  if (ivalue_python.isInt() && ivalue_cpp.isIntList()) {
    auto ivalue_cpp_list = ivalue_cpp.toIntListRef();
    std::vector<int64_t> ivalue_python_vec(ivalue_cpp_list.size());
    std::fill(ivalue_python_vec.begin(), ivalue_python_vec.end(), ivalue_python.toInt());
    return ivalue_python_vec == ivalue_cpp_list;
  }

  // For Python modules, we allow the use of "none" / "mean" / "sum" to represent the reduction type.
  // The corresponding data type for C++ modules is `torch::Reduction::Reduction` enum, and here we map the
  // reduction types between Python version and C++ version.
  if (ivalue_python.isString() && ivalue_cpp.isInt()) {
    auto& ivalue_python_str = ivalue_python.toStringRef();
    auto ivalue_cpp_int = ivalue_cpp.toInt();
    if (ivalue_python_str == "none") {
      return ivalue_cpp_int == torch::Reduction::None;
    } else if (ivalue_python_str == "mean") {
      return ivalue_cpp_int == torch::Reduction::Mean;
    } else if (ivalue_python_str == "sum") {
      return ivalue_cpp_int == torch::Reduction::Sum;
    }
  }

  if (ivalue_python.tagKind() != ivalue_cpp.tagKind()) {
    AT_ERROR("Value type mismatch: ", "from Python: ", ivalue_python.tagKind(), ", from C++: ", ivalue_cpp.tagKind());
  }

  if (ivalue_python.isInt()) {
    return ivalue_python.toInt() == ivalue_cpp.toInt();
  } else if (ivalue_python.isDouble()) {
    return ivalue_python.toDouble() == ivalue_cpp.toDouble();
  } else if (ivalue_python.isBool()) {
    return ivalue_python.toBool() == ivalue_cpp.toBool();
  } else if (ivalue_python.isString()) {
    return ivalue_python.toStringRef() == ivalue_cpp.toStringRef();
  } else if (ivalue_python.isTensor()) {
    return check_tensor_equality(ivalue_python.toTensor(), ivalue_cpp.toTensor());
  } else if (ivalue_python.isIntList()) {
    return ivalue_python.toIntListRef() == ivalue_cpp.toIntListRef();
  } else if (ivalue_python.isNone()) {
    return ivalue_cpp.isNone();
  } else {
    AT_ERROR("Unsupported value type: ", ivalue_python.tagKind());
  }
}
"""

CHECK_MODULE_PARAM_EQUALITY = Template("""\
TORCH_CHECK(
  check_tensor_equality(${script_module_prefix}.get_parameter("${param_name}"), ${cpp_module_prefix}->${param_name}),
  GENERATE_PARITY_TEST_ERROR_MSG(
    "`${cpp_module_prefix}->${param_name}`",
    ${cpp_module_prefix}->${param_name},
    ${script_module_prefix}.get_parameter("${param_name}")));
TORCH_CHECK(
  ${script_module_prefix}.get_parameter("${param_name}").requires_grad() == ${cpp_module_prefix}->${param_name}.requires_grad(),
  GENERATE_PARITY_TEST_ERROR_MSG(
    "`${cpp_module_prefix}->${param_name}.requires_grad()`",
    ${cpp_module_prefix}->${param_name}.requires_grad(),
    ${script_module_prefix}.get_parameter("${param_name}").requires_grad()));
""")

CHECK_MODULE_BUFFER_EQUALITY = Template("""\
TORCH_CHECK(
  check_tensor_equality(${script_module_prefix}.get_buffer("${buffer_name}"), ${cpp_module_prefix}->${buffer_name}),
  GENERATE_PARITY_TEST_ERROR_MSG(
    "`${cpp_module_prefix}->${buffer_name}`",
    ${cpp_module_prefix}->${buffer_name},
    ${script_module_prefix}.get_buffer("${buffer_name}")));
""")

CHECK_MODULE_ATTR_EQUALITY = Template("""\
TORCH_CHECK(
  check_ivalue_equality(
    ${script_module_prefix}.get_attribute("${python_attr_name}"), c10::IValue(${cpp_module_prefix}->${cpp_attr_name})),
  GENERATE_PARITY_TEST_ERROR_MSG(
    "`${cpp_module_prefix}->${cpp_attr_name}`",
    c10::IValue(${cpp_module_prefix}->${cpp_attr_name}),
    ${script_module_prefix}.get_attribute("${python_attr_name}")));
""")

TORCH_NN_MODULE_TEST_CTOR_ARGS = Template("""\n
void ${module_name}_test_ctor_args() {
  ${module_qualified_name} m_init_by_cpp(${module_option});

  ${extra_stmts}
}
""")

TORCH_NN_MODULE_TEST_OPTIONS_ARG = Template("""\
m_init_by_cpp->options.${options_arg_name}();
""")

TORCH_NN_MODULE_TEST_INIT = Template("""\n
void ${module_variant_name}_test_init(
    const std::string& saved_module_path,
    const std::string& device) {
  torch::jit::script::Module m_init_by_python = torch::jit::load(saved_module_path);

  torch::manual_seed(2);
  ${module_qualified_name} m_init_by_cpp${cpp_constructor_args};
  m_init_by_cpp->to(device);

  ${extra_stmts}
}
""")

TORCH_NN_MODULE_TEST_FORWARD = Template("""\n
void ${module_variant_name}_test_forward(
    const std::string& saved_module_path,
    const std::string& device,
    torch::Tensor python_output,
    ${input_arg_declarations}) {
  torch::manual_seed(2);
  ${module_qualified_name} module${cpp_constructor_args};
  torch::load(module, saved_module_path);
  module->to(device);

  auto cpp_output = module(${input_args});

  TORCH_CHECK(
    check_tensor_equality(cpp_output, python_output),
    GENERATE_PARITY_TEST_ERROR_MSG(
      "forward output",
      cpp_output,
      python_output));

  ${extra_stmts}
}
""")

TORCH_NN_MODULE_TEST_BACKWARD = Template("""\n
void ${module_variant_name}_test_backward(
    const std::string& saved_module_path,
    const std::string& saved_grad_module_path,
    const std::string& device,
    ${input_arg_declarations}) {
  ${module_qualified_name} python_grad_module${cpp_constructor_args};
  torch::load(python_grad_module, saved_grad_module_path);

  torch::manual_seed(2);
  ${module_qualified_name} module${cpp_constructor_args};
  torch::load(module, saved_module_path);
  module->to(device);

  auto cpp_output = module(${input_args});
  cpp_output.sum().backward();

  for (size_t i = 0; i < module->parameters().size(); i++) {
    auto named_param = module->named_parameters()[i];
    auto grad = python_grad_module->parameters()[i];
    TORCH_CHECK(
      check_tensor_equality(named_param->grad(), grad),
      GENERATE_PARITY_TEST_ERROR_MSG(
        "gradient of `" + named_param.key() + "`",
        named_param->grad(),
        grad));
  }

  ${extra_stmts}
}
""")

TORCH_NN_MODULE_IGNORED_ATTRS = {
    '_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks',
    '_state_dict_hooks', '_load_state_dict_pre_hooks', '_modules', 'training',
}

# yf225 TODO: write doc for this function
def _get_python_module_init_arg_spec(module_name):
    python_module_class = getattr(torch.nn, module_name)
    if PY2:
        init_arg_spec = inspect.getargspec(python_module_class.__init__)
    else:
        init_arg_spec = inspect.getfullargspec(python_module_class.__init__)
    return init_arg_spec

# yf225 TODO: write doc for this function
def _prepare_tensors_for_module_input_or_target(test_params, tensors):
    if type(tensors) == tuple:
        tensors = list(tensors)
    elif type(tensors) == torch.Tensor:
        tensors = [tensors]
    else:
        raise RuntimeError("Unexpected input type: {}".format(type(tensors)))

    if test_params.device != 'cuda' or TEST_CUDA:
        tensors = [x.to(test_params.device) for x in tensors]

    return tensors

# yf225 TODO: write doc for this function
def _get_example_inputs(test_params):
    example_inputs = test_params.test_instance._get_input()
    example_inputs = _prepare_tensors_for_module_input_or_target(test_params, example_inputs)

    # We set all inputs to torch.nn module to requires grad, so that the backward test can always be run.
    # However, we skip embedding layers for now, because they only accept LongTensor as inputs,
    # And LongTensor cannot require grad.
    if test_params.module_name not in ["Embedding", "Embedding_sparse", "EmbeddingBag", "EmbeddingBag_sparse"]:
        example_inputs = [x.requires_grad_() for x in example_inputs]

    return example_inputs

# yf225 TODO: write doc for this function
def _get_example_targets(self, test_params):
    example_targets = test_params.test_instance._get_target()
    example_targets = _prepare_tensors_for_module_input_or_target(test_params, example_targets)
    return example_targets

# yf225 TODO: write doc for this function
def _get_forward_input_args(self, test_params):
    example_inputs = _get_example_inputs(test_params)
    if isinstance(test_params.test_instance, common_nn.CriterionTest):
        example_targets = _get_example_targets(test_params)
    else:
        example_targets = []

    input_args = ()
    for example_input in example_inputs:
        input_args += (example_input, )
    for example_target in example_targets:
        input_args += (example_target, )

    return input_args

# yf225 TODO: write doc for this function
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

# yf225 TODO: write doc for this function
def _process_test_params(test_params_dict, module_metadata, device, is_criterion):
    module_name = _compute_module_name(test_params_dict)
    test_params_dict['constructor'] = test_params_dict.get('constructor', getattr(torch.nn, module_name))
    if is_criterion:
        test = common_nn.CriterionTest(**test_params_dict)
    else:
        test = common_nn.ModuleTest(**test_params_dict)
    module_variant_name = test.get_name()[5:] + (('_' + device) if device != 'cpu' else '')

    return TorchNNTestParams(
        module_name=module_name,
        module_variant_name=module_variant_name,
        test_instance=test,
        cpp_constructor_args=test_params_dict.get('cpp_constructor_args'),
        has_parity=test_params_dict.get('has_parity', True),
        device=device,
    )