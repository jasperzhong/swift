import os
import tempfile
from string import Template
import copy
import unittest
import warnings
import inspect
import re

import torch
from torch._six import PY2
import common_utils as common
import common_nn
from common_cuda import TEST_CUDA
import torch.utils.cpp_extension
from cpp_api_parity import sample_module, torch_nn_module_configs, TorchNNTestParams, CppArg, parse_parity_tracker_table
from cpp_api_parity.common_test_harness import _python_arg_to_cpp_arg, _compile_cpp_code_inline
from cpp_api_parity.torch_nn_module_test_harness import TORCH_NN_MODULE_COMMON_TEST_HARNESS, CHECK_MODULE_PARAM_EQUALITY, \
  CHECK_MODULE_BUFFER_EQUALITY, CHECK_MODULE_ATTR_EQUALITY, TORCH_NN_MODULE_TEST_CTOR_ARGS, TORCH_NN_MODULE_TEST_OPTIONS_ARG, \
  TORCH_NN_MODULE_TEST_INIT, TORCH_NN_MODULE_TEST_FORWARD, TORCH_NN_MODULE_TEST_BACKWARD, TORCH_NN_MODULE_IGNORED_ATTRS, \
  _get_python_module_init_arg_spec, _prepare_tensors_for_module_input_or_target, _get_example_inputs, _get_example_targets

# yf225 TODO: lay out the overall strategy of how we do C++/Python parity test here!

# yf225 TODO: encapsulate and hide the complexities, and add comment to explain the complexities

parity_table_path = os.path.join(os.path.dirname(__file__), 'cpp_api_parity/parity-tracker.md')

parity_table = parse_parity_tracker_table(parity_table_path)

# yf225 TODO: less confusion about what each function does and what something means, and more encapsulation!!
# yf225 TODO: move all utilities function to a separate file, and keep functions here that absolutely need to be here.

class TestCppApiParity(common.TestCase):
    # This tests that Python and C++ torch.nn modules have matching constructor arg names and types.
    def _test_torch_nn_module_ctor_args(self, module_name):
        module_metadata = torch_nn_modules.module_metadata_map[module_name]
        cpp_default_constructor_args_str = module_metadata.cpp_default_constructor_args
        init_arg_spec = _get_python_module_init_arg_spec(module_name)
        init_kwargs_defaults = init_arg_spec.defaults
        python_default_constructor_arg_names = [
            x for x in init_arg_spec.args[1:-len(init_kwargs_defaults)]
            if x not in module_metadata.python_ignored_constructor_args]
        # NOTE: the regex is used here to split up e.g. `(1, {2, 3}, 4)` into `['1', '{2, 3}', '4']`
        cpp_default_constructor_arg_values = re.findall(r'{[^}]*}|[^,\s()]+', cpp_default_constructor_args_str)

        # Step 1: Check that the # of non-keyword args in C++ module constructor is equal to that in Python module constructor.
        self.assertEqual(
            len(cpp_default_constructor_arg_values),
            len(python_default_constructor_arg_names),
            "The constructor of `torch::nn::{}` in C++ ".format(module_name) +
            "must take the exact same number of non-keyword arguments " +
            "as the constructor of `torch.nn.{}` in Python. ".format(module_name) +
            "However, currently the C++ constructor expects {} non-keyword argument(s) ".format(
                len(cpp_default_constructor_arg_values)) +
            "while the Python constructor expects {} non-keyword argument(s): {}".format(
                len(python_default_constructor_arg_names),
                python_default_constructor_arg_names))

        # Step 2: Generate code to construct C++ module options using values from `cpp_default_constructor_args`.
        cpp_module_option = 'torch::nn::{}Options{}'.format(module_name, cpp_default_constructor_args_str)
        init_kwargs = init_arg_spec.args[-len(init_kwargs_defaults):]
        for arg_name, python_default_value in zip(init_kwargs, init_kwargs_defaults):
            # NOTE: If a Python module constructor arg's default value is None, we don't test its corresponding
            # options arg in C++ module (because the way to set the C++ options arg to an empty value is to not
            # specify it, which means we can't test that the options arg exists).
            # Instead, we test that all options args exist by calling their accessors after constructing the
            # C++ module with the options.
            if arg_name not in module_metadata.python_ignored_constructor_args and python_default_value is not None:
                cpp_module_option += '.{}({})'.format(arg_name, _python_arg_to_cpp_arg(python_default_value).value)

        # Step 3: Generate code to check existence of all Python module constructor args in the C++ module options.
        extra_stmts = [TORCH_NN_MODULE_TEST_OPTIONS_ARG.substitute(options_arg_name=arg_name)
                       for arg_name in python_default_constructor_arg_names + init_kwargs
                       if arg_name not in module_metadata.python_ignored_constructor_args]

        # Step 4: Compile the test code and run the tests.
        cpp_sources = TORCH_NN_MODULE_COMMON_TEST_HARNESS + module_metadata.cpp_sources
        cpp_sources += TORCH_NN_MODULE_TEST_CTOR_ARGS.substitute(
            module_name=module_name,
            module_qualified_name='torch::nn::{}'.format(module_name),
            module_option=cpp_module_option,
            extra_stmts=''.join(extra_stmts))
        cpp_test_name = module_name + '_test_ctor_args'
        cpp_module = _compile_cpp_code_inline(
            name=cpp_test_name, cpp_sources=cpp_sources, functions=cpp_test_name)

        getattr(cpp_module, cpp_test_name)()

    # yf225 TODO: write doc for this function
    def _test_torch_nn_module_variant(self, test_params):
        def get_python_ignored_attrs(module_metadata):
            return list(TORCH_NN_MODULE_IGNORED_ATTRS) + module_metadata.python_ignored_attrs

        def generate_test_cpp_sources(test_params, template, extra_stmts):
            input_args = self._get_forward_input_args(test_params)
            input_arg_types = [_python_arg_to_cpp_arg(arg).type for arg in list(input_args)]
            input_args = ['arg{}'.format(str(i)) for i in range(len(input_arg_types))]
            input_arg_declarations = ['{} {}'.format(arg_type, arg_name) for arg_type, arg_name in zip(input_arg_types, input_args)]
            test_cpp_sources = template.substitute(
                module_variant_name=test_params.module_variant_name,
                module_qualified_name='torch::nn::{}'.format(test_params.module_name),
                cpp_constructor_args=test_params.cpp_constructor_args,
                input_arg_declarations=',\n'.join(input_arg_declarations),
                input_args=',\n'.join(input_args),
                extra_stmts=extra_stmts)
            return test_cpp_sources

        def setup_init_test(test_params):
            module_metadata = torch_nn_modules.module_metadata_map[test_params.module_name]

            # We are generating the attribute equality checks manually here,
            # because it is not possible to have a `.attributes()` API that returns
            # non-parameter / non-buffer attributes in a C++ torch::nn module.
            def generate_attr_equality_checks(module,
                                              script_module_prefix='m_init_by_python',
                                              cpp_module_prefix='m_init_by_cpp'):
                stmts = []
                for name, sub_module in module.named_children():
                    sub_script_module_prefix = '{}.get_module("{}")'.format(script_module_prefix, name)
                    sub_cpp_module_prefix = '{}->{}'.format(cpp_module_prefix, name)
                    stmts = generate_attr_equality_checks(sub_module, sub_script_module_prefix, sub_cpp_module_prefix)
                for name, param in module._parameters.items():
                    stmts.append(CHECK_MODULE_PARAM_EQUALITY.substitute(
                        script_module_prefix=script_module_prefix,
                        cpp_module_prefix=cpp_module_prefix,
                        param_name=name))
                for name, buffer in module._buffers.items():
                    stmts.append(CHECK_MODULE_BUFFER_EQUALITY.substitute(
                        script_module_prefix=script_module_prefix,
                        cpp_module_prefix=cpp_module_prefix,
                        buffer_name=name))

                init_arg_spec = _get_python_module_init_arg_spec(module.__class__.__name__)
                # NOTE: `init_arg_spec.args[0]` is `self`, which is not counted as a constructor arg in the API parity test.
                python_constructor_arg_names = [
                    x for x in init_arg_spec.args[1:] if x not in module_metadata.python_ignored_constructor_args]
                for name, attr in module.__dict__.items():
                    if name not in get_python_ignored_attrs(module_metadata):
                        # Every constructor arg of the Python module must have
                        # a corresponding C++ module options arg.
                        if name in python_constructor_arg_names:
                            cpp_attr_name = 'options.{}()'.format(name)
                        else:
                            cpp_attr_name = name
                        stmts.append(CHECK_MODULE_ATTR_EQUALITY.substitute(
                            script_module_prefix=script_module_prefix,
                            cpp_module_prefix=cpp_module_prefix,
                            python_attr_name=name,
                            cpp_attr_name=cpp_attr_name))
                return stmts

            device = test_params.device
            python_constructor = test_params.test_instance.constructor
            python_constructor_args = test_params.test_instance.constructor_args

            torch.manual_seed(2)
            module = python_constructor(*python_constructor_args).to(device)

            extra_stmts = generate_attr_equality_checks(module)
            assert len(extra_stmts) == module_metadata.num_attrs_recursive
            extra_stmts_str = ''.join(extra_stmts)
            return (([module], device),
                    generate_test_cpp_sources(
                        test_params=test_params, template=TORCH_NN_MODULE_TEST_INIT, extra_stmts=extra_stmts_str))

        def setup_forward_test(test_params):
            device = test_params.device
            python_constructor = test_params.test_instance.constructor
            python_constructor_args = test_params.test_instance.constructor_args
            input_args = self._get_forward_input_args(test_params)

            torch.manual_seed(2)
            module = python_constructor(*python_constructor_args).to(device)
            python_output = module(*input_args)

            return (([module], device, python_output, input_args),
                    generate_test_cpp_sources(
                        test_params=test_params, template=TORCH_NN_MODULE_TEST_FORWARD, extra_stmts=''))

        def setup_backward_test(test_params):
            device = test_params.device
            python_constructor = test_params.test_instance.constructor
            python_constructor_args = test_params.test_instance.constructor_args
            input_args = self._get_forward_input_args(test_params)

            torch.manual_seed(2)
            module = python_constructor(*python_constructor_args).to(device)
            python_output = module(*input_args)
            python_output.sum().backward()
            # JIT tracing does not save a module's parameters' gradients into ScriptModule.
            # Instead, we create another module `grad_module` with the same structure as `module`,
            # and use `grad_module`'s parameters to save `module`'s corresponding parameters'
            # gradients. Then, we trace both `module` and `grad_module`, serialize them and
            # pass them into C++ for parity testing.
            grad_module = copy.deepcopy(module)
            for param, grad_param in zip(module.parameters(), grad_module.parameters()):
                if param.grad is not None:
                    grad_param.data = param.grad

            return (([module, grad_module], device, input_args),
                    generate_test_cpp_sources(
                        test_params=test_params, template=TORCH_NN_MODULE_TEST_BACKWARD, extra_stmts=''))

        def trace_module(module, input_args):
            module_metadata = torch_nn_modules.module_metadata_map[module.__class__.__name__]

            # JIT tracing does not automatically save a module's non-parameter / non-buffer attributes
            # into a ScriptModule's slots, which means we can't access them via `get_attributes()` in C++.
            # Here, we manually register these attributes into the ScriptModule so that we can access them
            # via `get_attributes()` in C++.
            def register_attrs(module, script_module):
                for sub_module, sub_script_module in zip(module.children(), script_module.children()):
                    register_attrs(sub_module, sub_script_module)
                for key, value in module.__dict__.items():
                    if key not in get_python_ignored_attrs(module_metadata):
                        if value is None:
                            value_type = module_metadata.python_optional_attribute_to_jit_type[key]
                        elif type(value) == tuple:
                            assert all(isinstance(x, type(value[0])) for x in value), \
                                "All elements in a tuple attribute of a Python torch.nn module must have the same type."
                            # Here, we set the Python tuple attribute's type to `ListType` in the ScriptModule,
                            # which will automatically be converted to `IntList` later and match the type
                            # of the corresponding attribute in C++ module (which is initially an `ExpandingArray`
                            # and is converted to `IntList` by the `IValue` constructor).
                            value_type = torch._C.ListType(torch.jit.annotations.ann_to_type(type(value[0])))
                        else:
                            value_type = torch.jit.annotations.ann_to_type(type(value))
                        script_module._c._register_attribute(key, value_type, value)

            # We use JIT tracing to serialize Python module state, so that we can load it into C++
            traced_script_module = torch.jit.trace(module, input_args)
            register_attrs(module, traced_script_module)
            return traced_script_module

        def serialize_module_into_file(script_module):
            module_file = tempfile.NamedTemporaryFile(delete=False)
            script_module.save(module_file.name)
            module_file.close()
            return module_file.name

        def test_methods(test_params):
            module_metadata = torch_nn_modules.module_metadata_map[test_params.module_name]
            module_variant_name = test_params.module_variant_name
            input_args = self._get_forward_input_args(test_params)

            args_map = {}

            cpp_sources = TORCH_NN_MODULE_COMMON_TEST_HARNESS + module_metadata.cpp_sources

            torch_nn_test_methods = [
                ('init', setup_init_test),
                ('forward', setup_forward_test),
                ('backward', setup_backward_test),
            ]
            for method_name, setup_test in torch_nn_test_methods:
                args_map[method_name], test_cpp_sources = setup_test(test_params)
                cpp_sources += test_cpp_sources

            cpp_module = _compile_cpp_code_inline(
                name=test_params.module_variant_name,
                cpp_sources=cpp_sources,
                functions=['{}_test_{}'.format(
                    test_params.module_variant_name,
                    method_name) for method_name, _ in torch_nn_test_methods])

            for method_name, _ in torch_nn_test_methods:
                args = args_map[method_name]
                modules = args[0]
                script_modules = [trace_module(module, input_args) for module in modules]
                module_file_names = [serialize_module_into_file(script_module) for script_module in script_modules]

                cpp_args = module_file_names[:]
                for arg in args[1:]:
                    if isinstance(arg, tuple):
                        cpp_args += list(arg)
                    elif isinstance(arg, list):
                        cpp_args += arg
                    else:
                        cpp_args.append(arg)

                try:
                    cpp_test_name = '{}_test_{}'.format(module_variant_name, method_name)
                    cpp_test_fn = getattr(cpp_module, cpp_test_name)
                    if not test_params.has_parity:
                        with self.assertRaisesRegex(RuntimeError, "Parity test failed"):
                            cpp_test_fn(*cpp_args)
                    else:
                        cpp_test_fn(*cpp_args)
                finally:
                    # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
                    # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
                    # we close the file after creation and try to remove it manually.
                    for module_file_name in module_file_names:
                        try:
                            os.remove(module_file_name)
                        except OSError as e:
                            warnings.warn("Unable to remove {}, got error: {}".format(module_file_name, str(e)))

        test_methods(test_params)


def has_test(test_name):
    return hasattr(TestCppApiParity, test_name)

def add_test(test_name, test_fn):
    if has_test(test_name):
        raise RuntimeError("Found two tests with the same name: " + test_name)
    setattr(TestCppApiParity, test_name, test_fn)

devices = ['cpu', 'cuda']

torch_nn_test_params_map = {}


def add_torch_nn_module_tests(module_tests, is_criterion):
    for test_params_dict in module_tests:
        # We skip all `torch.nn.functional` tests for now
        if 'FunctionalModule' in str(test_params_dict.get('constructor', '')):
            continue

        module_name = _compute_module_name(test_params_dict)

        assert hasattr(torch.nn, module_name), \
            "`torch.nn` doesn't have module `{}`. ".format(module_name) + \
            "If you are adding a new test, please set `fullname` using format `ModuleName_desc`, " + \
            "or set `module_name` using format `ModuleName`."

        module_full_name = 'torch.nn.' + module_name
        if module_full_name not in parity_table['torch.nn']:
            raise RuntimeError(
                'Module `{}` is not found in Python / C++ API parity table. Please update parity table at {}.'.format(
                    module_full_name, parity_table_path))

        has_impl_parity, _ = parity_table['torch.nn'][module_full_name]

        def add_ctor_args_test_for_module(module_name, has_impl_parity):
            ctor_args_test_name = 'test_torch_nn_{}_ctor_args'.format(module_name)

            def ctor_args_test(self):
                self._test_torch_nn_module_ctor_args(
                    module_name=self._testMethodName.replace('test_torch_nn_', '').replace('_ctor_args', ''))

            if not has_impl_parity:
                ctor_args_test = unittest.expectedFailure(ctor_args_test)

            # We only run one constructor args test per module
            if not has_test(ctor_args_test_name):
                add_test(ctor_args_test_name, ctor_args_test)

        def add_variant_test_for_module(module_name, test_params_dict, has_impl_parity):
            module_metadata = torch_nn_modules.module_metadata_map[module_name]
            for device in devices:
                test_params = _process_test_params(
                    test_params_dict=test_params_dict,
                    module_metadata=module_metadata,
                    device=device,
                    is_criterion=is_criterion)
                test_name = 'test_torch_nn_{}'.format(test_params.module_variant_name)
                torch_nn_test_params_map[test_name] = test_params

                def test_fn(self):
                    self._test_torch_nn_module_variant(test_params=torch_nn_test_params_map[self._testMethodName])

                if device == 'cuda':
                    test_fn = unittest.skipIf(not TEST_CUDA, "CUDA unavailable")(test_fn)

                if not has_impl_parity:
                    test_fn = unittest.expectedFailure(test_fn)

                add_test(test_name, test_fn)

        add_ctor_args_test_for_module(module_name, has_impl_parity)
        add_variant_test_for_module(module_name, test_params_dict, has_impl_parity)

add_torch_nn_module_tests(
    sample_module.module_tests + common_nn.module_tests + common_nn.new_module_tests,
    is_criterion=False)

add_torch_nn_module_tests(
    common_nn.criterion_tests + common_nn.new_criterion_tests,
    is_criterion=True)

# yf225 TODO: instead of asserting that tests for SampleModule exists, we should check in the generated test code and eye-ball it.
# and we add test to ensure the generated test code is always the same as the checked-in code.
# yf225 TODO: we should combine all the checked-in test code in a Python file, something like:
# ```
# torch_nn_Linear_variant1_test_code = "..."
# torch_nn_BatchNorm1d_variant1_test_code = "..."
# ```
#
# Assert that there exists auto-generated tests for SampleModule.
assert len([name for name in TestCppApiParity.__dict__ if 'SampleModule' in name]) == \
    len(sample_module.module_tests) * len(devices) + 1


if __name__ == "__main__":
    common.run_tests()
