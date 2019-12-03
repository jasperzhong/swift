from cpp_api_parity import CppArg

# yf225 TODO: write doc for this function
def _python_arg_to_cpp_arg(python_arg):
    if type(python_arg) == int:
        return CppArg(type='int64_t', value=str(python_arg))
    elif type(python_arg) == float:
        return CppArg(type='double', value=str(python_arg))
    elif type(python_arg) == bool:
        return CppArg(type='bool', value=str(python_arg).lower())
    elif type(python_arg) == str:
        # if `python_arg` is one of the reduction types, we use the corresponding `torch::kXXX` enum.
        if python_arg in ['none', 'mean', 'sum']:
            if python_arg == 'none':
                cpp_arg = 'torch::kNone'
            elif python_arg == 'mean':
                cpp_arg = 'torch::kMean'
            elif python_arg == 'sum':
                cpp_arg = 'torch::kSum'
            return CppArg(type='c10::variant', value='{}'.format(cpp_arg))
        else:
            raise Exception
    elif type(python_arg) == torch.Tensor:
        return CppArg(
            type='torch::Tensor',
            value='torch::empty({})'.format(str(list(python_arg.shape)).replace('[', '{').replace(']', '}')))
    else:
        raise RuntimeError(
            "{} is not a supported arg type for C++ module methods".format(type(python_arg)))

# yf225 TODO: write doc for this function
def _compile_cpp_code_inline(name, cpp_sources, functions):
    # Just-in-time compile the C++ test code
    cpp_module = torch.utils.cpp_extension.load_inline(
        name=name,
        cpp_sources=cpp_sources,
        functions=functions,
        verbose=False,
    )
    return cpp_module