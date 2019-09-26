# Generates C++ functions that wrap ATen tensor factory methods to turn them into Variables.
#
# This writes one file: variable_factories.h

import re

from .utils import CodeTemplate, write
from .gen_variable_type import format_trace

# [CHECK THIS] this is a hack. should be fixed later on
FUNCTION_TEMPLATE_ARANGE = CodeTemplate("""\
inline at::Tensor ${name}(${collapsed_formals}) {
  ${pre_record_trace}
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    auto currDtype = options.dtype_opt();
    if (currDtype.has_value()) {
      return at::_arange(${uncollapsed_actuals});
    }
    return at::_arange(${uncollapsed_actuals_nullptr});
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  ${post_record_trace}
  return result;
}
""")

FUNCTION_TEMPLATE_TENSOR_OPTIONS = CodeTemplate("""\
inline at::Tensor ${name}(${collapsed_formals}) {
  ${pre_record_trace}
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_${name}(${uncollapsed_actuals});
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/options.requires_grad());
  ${post_record_trace}
  return result;
}
""")

FUNCTION_TEMPLATE = CodeTemplate("""\
inline at::Tensor ${name}(${formals}) {
  ${pre_record_trace}
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_${name}(${actuals});
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/${requires_grad});
  ${post_record_trace}
  return result;
}
""")

TYPE_PATTERN = re.compile(r"(?:const\s+)?([A-Z]\w+)")

def check_if_factory_method(args):
    a = any(arg['type'] == 'c10::optional<ScalarType>' for arg in args) and any(arg['type'] == 'c10::optional<Layout>' for arg in args) and any(arg['type'] == 'c10::optional<Device>' for arg in args) and any(arg['type'] == 'c10::optional<bool>' for arg in args)
    c = any(arg['type'] == 'ScalarType' for arg in args) and any(arg['type'] == 'Layout' for arg in args) and any(arg['type'] == 'Device' for arg in args) and any(arg['type'] == 'bool' for arg in args)
    b = any('TensorOptions' in arg['type'] for arg in args)
    return a or b or c
    
def fully_qualified_type(argument_type):
    match = TYPE_PATTERN.match(argument_type)
    if match is None:
        return argument_type
    index = match.start(1)
    return "{}at::{}".format(argument_type[:index], argument_type[index:])


def gen_variable_factories(out, declarations, template_path, disable_autograd=False):
    function_definitions = []
    for decl in declarations:
        has_tensor_options = check_if_factory_method(decl["arguments"])        
        is_namespace_fn = 'namespace' in decl['method_of']
        if (has_tensor_options or decl["name"].endswith("_like")) and is_namespace_fn:
            function_definitions.append(
                process_function(decl, has_tensor_options, disable_autograd=disable_autograd))
    write(out,
          "variable_factories.h",
          CodeTemplate.from_file(template_path + "/variable_factories.h"),
          {"function_definitions": function_definitions})

def collapse_formals(formals):
        collapsed = formals.copy()
        if (any(formal == 'c10::optional<ScalarType> dtype' for formal in formals) and
            any(formal == 'c10::optional<Layout> layout' for formal in formals) and
            any(formal == 'c10::optional<Device> device' for formal in formals) and 
            any(formal == 'c10::optional<bool> pin_memory' for formal in formals)):
            index = formals.index('c10::optional<ScalarType> dtype')

            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.insert(index, 'const at::TensorOptions & options /*[CHECK THIS] should have ={}*/')

        if ((any(formal == 'c10::optional<ScalarType> dtype = c10::nullopt' for formal in formals) or any(formal == 'c10::optional<ScalarType> dtype = at::kLong' for formal in formals)) and
            any(formal == 'c10::optional<Layout> layout = c10::nullopt' for formal in formals) and
            any(formal == 'c10::optional<Device> device = c10::nullopt' for formal in formals) and 
            any(formal == 'c10::optional<bool> pin_memory = c10::nullopt' for formal in formals)):
            if 'c10::optional<ScalarType> dtype = c10::nullopt' in formals:
                index = formals.index('c10::optional<ScalarType> dtype = c10::nullopt')
            else:
                index = formals.index('c10::optional<ScalarType> dtype = at::kLong')

            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.insert(index, 'const at::TensorOptions & options={}')

        if (any(formal == 'at::ScalarType dtype' for formal in formals) and
            any(formal == 'at::Layout layout' for formal in formals) and
            any(formal == 'at::Device device' for formal in formals) and 
            (any(formal == 'bool pin_memory' for formal in formals) or any(formal == 'bool pin_memory = false' for formal in formals))):
            index = formals.index('at::ScalarType dtype')

            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.pop(index)
            collapsed.insert(index, 'const at::TensorOptions & options')
        
        return collapsed

def collapse_actuals(actuals):
    collapsed = actuals.copy()
    index = actuals.index('dtype')
    collapsed[index] = 'at::typeMetaToScalarType(options.dtype())'
    collapsed[index + 1] = 'options.layout()'
    collapsed[index + 2] = 'options.device()'
    collapsed[index + 3] = 'options.pinned_memory()'
    return collapsed

def replace_dtype_nullprt(actuals):
    replaced = actuals.copy()
    index = actuals.index('at::typeMetaToScalarType(options.dtype())')
    replaced[index] = 'c10::nullopt'
    return replaced
    
def process_function(decl, has_tensor_options, disable_autograd):
    formals = []
    actuals = []
    for argument in decl["arguments"]:
        type = fully_qualified_type(argument["type"])
        
        default = " = {}".format(argument["default"]) if "default" in argument else ""
        if "default" in argument:
            if argument["default"] == False or argument["default"] == True:
                default = default.lower()

        formals.append("{} {}{}".format(type, argument["name"], default))

        actual = argument["name"]
        if argument["simple_type"] == "TensorOptions":
            actual = "at::TensorOptions({})".format(actual)
        actuals.append(actual)

    requires_grad = "false"
    if decl['name'].endswith('_like') and not has_tensor_options:
        # Insert TensorOptions before MemoryFormat
        actuals.insert(-1, 'at::typeMetaToScalarType({}.options().dtype())'.format(actuals[0]))
        actuals.insert(-1, '{}.options().layout()'.format(actuals[0]))
        actuals.insert(-1, '{}.options().device()'.format(actuals[0]))
        actuals.insert(-1, '{}.options().pinned_memory()'.format(actuals[0]))
        
    if not disable_autograd:
        pre_record_trace, post_record_trace = format_trace(decl)
        if has_tensor_options:
            pre_record_trace = pre_record_trace.replace("jit::tracer::addInputs(node, \"dtype\", dtype);",
                                                        "jit::tracer::addInputs(node, \"options\", options);")
            pre_record_trace = pre_record_trace.replace("jit::tracer::addInputs(node, \"layout\", layout);",
                                                        "")
            pre_record_trace = pre_record_trace.replace("jit::tracer::addInputs(node, \"device\", device);",
                                                        "")
            pre_record_trace = pre_record_trace.replace("jit::tracer::addInputs(node, \"pin_memory\", pin_memory);",
                                                        "")
    else:
        pre_record_trace, post_record_trace = '', ''

    if not has_tensor_options:
        return FUNCTION_TEMPLATE.substitute(
            name=decl["name"], formals=formals, actuals=actuals, requires_grad=requires_grad,
            pre_record_trace=pre_record_trace, post_record_trace=post_record_trace
        )
    else:
        uncollapsed_actuals = collapse_actuals(actuals)
        collapsed_formals = collapse_formals(formals)
        uncollapsed_actuals_nullptr = replace_dtype_nullprt(uncollapsed_actuals)

        if decl['name'] == 'arange':
            return FUNCTION_TEMPLATE_ARANGE.substitute(
                name=decl["name"], collapsed_formals = collapsed_formals, actuals=actuals, uncollapsed_actuals=uncollapsed_actuals, uncollapsed_actuals_nullptr=uncollapsed_actuals_nullptr, requires_grad=requires_grad,
                pre_record_trace=pre_record_trace, post_record_trace=post_record_trace
            )
        else:
            return FUNCTION_TEMPLATE_TENSOR_OPTIONS.substitute(
                name=decl["name"], collapsed_formals = collapsed_formals, actuals=actuals, uncollapsed_actuals=uncollapsed_actuals, requires_grad=requires_grad,
                pre_record_trace=pre_record_trace, post_record_trace=post_record_trace
            )