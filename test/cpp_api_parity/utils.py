from collections import namedtuple

TorchNNTestParams = namedtuple(
    'TorchNNTestParams',
    [
        'module_name',
        'module_variant_name',
        'test_instance',
        'cpp_constructor_args',
        'cpp_input_args',
        'cpp_input_args_requires_grad',
        'cpp_target_args',
        'cpp_extra_args',
        'has_parity',
        'device',
        'cpp_output_tmp_folder',
    ]
)

CppArg = namedtuple('CppArg', ['type', 'value'])
