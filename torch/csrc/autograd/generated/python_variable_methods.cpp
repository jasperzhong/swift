// @generated from tools/autograd/templates/python_variable_methods.cpp

#include <Python.h>

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Size.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/autograd/utils/error_messages.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/jit/tracer.h"
#ifdef USE_CUDA
#include "torch/csrc/cuda/Stream.h"
#include "torch/csrc/cuda/Event.h"
#endif
#include "torch/csrc/utils/cuda_lazy_init.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/python_tuples.h"
#include "torch/csrc/utils/tensor_apply.h"
#include "torch/csrc/utils/tensor_list.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/utils/tensor_types.h"
#include "torch/csrc/utils/structseq.h"

#include <ATen/ATen.h>
#include "c10/util/Optional.h"

#include <stdexcept>

using at::DeviceGuard;
using at::device_of;
using at::OptionalDeviceGuard;
using at::Backend;
using at::Scalar;
using at::ScalarType;
using at::Tensor;
using at::Layout;
using at::Device;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

static PyObject * THPVariable__is_view(PyObject *self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.is_view()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc no support for first-class functions in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_apply_(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.requires_grad()) {
    throw std::runtime_error(
        "Can't call apply_() on Variable that requires grad. Use "
        "var.detach().apply_() instead.");
  }
  return THPVariable_Wrap(torch::utils::apply_(self_, arg));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_size(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "size(int64_t dim)",
    "size()",
    "size(Dimname dim)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (jit::tracer::isTracing()) {
      return wrap(jit::tracer::getSizeOf(self_, r.toInt64(0)));
    } else {
      return wrap(self_.size(r.toInt64(0)));
    }
  } else if (r.idx == 1) {
    // we can't do the normal wrapping here because IntArrayRef maps to both
    // torch.Size and tuple in python.
    return THPSize_New(self_);
  }
  else if (r.idx == 2) {
    if (jit::tracer::isTracing()) {
      TORCH_INTERNAL_ASSERT(false, "NYI: Named tensors w/ JIT");
    }
    return wrap(self_.size(r.dimname(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_stride(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "stride(int64_t dim)",
    "stride()",
    "stride(Dimname dim)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(self_.stride(r.toInt64(0)));
  } else if (r.idx == 1) {
    // yes, this is called strides in ATen.
    IntArrayRef strides = self_.strides();
    // we can't do the normal wrapping here because IntArrayRef maps to both
    // torch.Size and tuple in python
    return THPUtils_packInt64Array(strides.size(), strides.data());
  }
  else if (r.idx == 2) {
    return wrap(self_.stride(r.dimname(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_get_device(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(self.get_device());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_has_names(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(self.has_names());
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_data_ptr(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(self.data_ptr());
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_storage_offset(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(self.storage_offset());
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_dim(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
   return THPUtils_packInt64(self_.dim());
   END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_numel(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
   return THPUtils_packInt64(self_.numel());
   END_HANDLE_TH_ERRORS
}

static Tensor dispatch_contiguous(const Tensor & self, at::MemoryFormat memory_format) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.contiguous(memory_format);
}

static PyObject * THPVariable_contiguous(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "contiguous(*, MemoryFormat memory_format=contiguous_format)",
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  auto memory_format = r.memoryformat(0);
  // avoids touching the GIL or current device if self is already contiguous
  if (self_.is_contiguous(memory_format)) {
    // NOTE: this logic is duplicated from VariableType.cpp. Since we need to
    // record this call to contiguous() in the trace regardless of whether
    // we actually call contiguous here, we need to record this information
    // manually.
    if (jit::tracer::isTracing()) {
      auto tracer_state = jit::tracer::getTracingState();
      auto node = tracer_state->graph->create(jit::aten::contiguous, /*num_outputs=*/0);
      jit::tracer::recordSourceLocation(node);
      jit::tracer::addInputs(node, "self", self_);
      jit::tracer::addInputs(node, "memory_format", memory_format);
      tracer_state->graph->insertNode(node);
      jit::tracer::addOutput(node, self_);
    }
    Py_INCREF(self);
    return self;
  }
  return THPVariable_Wrap(dispatch_contiguous(self_, memory_format));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_copy_(Tensor & self, const Tensor & other, bool non_blocking) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.copy_(other, non_blocking);
}

 static PyObject * THPVariable_copy_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "copy_(Tensor other, bool non_blocking=False)",
    "copy_(Tensor other, bool async=False)|deprecated"
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  return THPVariable_Wrap(dispatch_copy_(self_, r.tensor(0), r.toBool(1)));
  END_HANDLE_TH_ERRORS
}

static double dispatch_to_CDouble(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.item<double>();
}

static std::complex<double> dispatch_to_CComplexDouble(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.item<std::complex<double>>();
}

static int64_t dispatch_to_CLong(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.item<int64_t>();
}

static bool dispatch_to_Bool(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.item<bool>();
}

static PyObject * THPVariable_float_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  jit::tracer::warn("Converting a tensor to a Python float", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_to_CDouble(self_));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_integral_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  jit::tracer::warn("Converting a tensor to a Python integer", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (isFloatingType(self_.scalar_type())) {
    // we can't dispatch to item<int64_t> here because we want to avoid ATen overflow checks;
    // the python integral type (long in python2) can't overflow.
    return THPUtils_packDoubleAsInt(dispatch_to_CDouble(self_));
  } else {
    return wrap(dispatch_to_CLong(self_));
  }
  END_HANDLE_TH_ERRORS
}

// This is the __index__ function in Python which is similar to __int__, but
// called when used as a slice.
static PyObject * THPVariable_index_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  jit::tracer::warn("Converting a tensor to a Python index", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  // TODO: change the condition to `self_.dim() != 0` once we expose scalars
  // in PyTorch.
  if (!isIntegralType(self_.scalar_type(), /*includeBool=*/true) || self_.numel() != 1) {
    throw TypeError("only integer tensors of a single element can be converted to an index");
  }
  return wrap(dispatch_to_CLong(self_));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_invert(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.bitwise_not();
}

static PyObject * THPVariable_invert(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (!isIntegralType(self_.scalar_type(), /*includeBool=*/true)) {
    throw TypeError("~ (operator.invert) is only implemented on integer and Boolean-type tensors");
  }
  return THPVariable_Wrap(dispatch_invert(self_));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_to(const Tensor & self, Device device, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // NOTE: this is where we record aten::to in the graph during tracing. However, the behavior of aten::to
  // is different with respect to TensorOptions fields that are not present: aten::to inherits fields that
  // are missing from the self argument while the tracer assumes that they should be populated with the
  // default values (eg. float for scalar type). By explicitly copying over the tensor options here we fully
  // specify all tensor options and thus record the proper trace
  return self.to(self.options().device(device), non_blocking, copy, optional_memory_format);
}

static Tensor dispatch_to(const Tensor & self, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  AutoNoGIL no_gil;
  return self.to(self.options(), non_blocking, copy, optional_memory_format);
}

static Tensor dispatch_to(const Tensor & self, ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  return self.to(dtype, non_blocking, copy, optional_memory_format);
}

static Tensor dispatch_to(const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  return self.to(device, dtype, non_blocking, copy, optional_memory_format);
}

static PyObject * THPVariable_cpu(PyObject* self, PyObject* args, PyObject* kwargs)
{
   HANDLE_TH_ERRORS
   static PythonArgParser parser({
     "cpu(*, MemoryFormat? memory_format=None)"
   });
   auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
   ParsedArgs<1> parsed_args;
   auto r = parser.parse(args, kwargs, parsed_args);
   auto opt_memory_format = r.memoryformatOptional(0);
   return THPVariable_Wrap(dispatch_to(self_, at::Device(at::DeviceType::CPU), false, false, opt_memory_format));
   END_HANDLE_TH_ERRORS
}

static Tensor dispatch_nonzero(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero();
}

static std::vector<Tensor> dispatch_nonzero_numpy(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero_numpy();
}

static PyObject * THPVariable_nonzero(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nonzero()|deprecated",
    "nonzero(*, bool as_tuple=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0 || (r.idx == 1 && !r.toBool(0))) {
    return wrap(dispatch_nonzero(self_));
  } else {
    return wrap(dispatch_nonzero_numpy(self_));
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_cuda(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cuda(Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "cuda(Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto device = r.isNone(0) ? at::Device(at::DeviceType::CUDA) : r.device(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  TORCH_CHECK(device.is_cuda(), "Invalid device, must be cuda device");
  torch::utils::cuda_lazy_init();
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_to_type(PyObject* self, ScalarType scalarType, c10::optional<c10::MemoryFormat> optional_memory_format) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return THPVariable_Wrap(dispatch_to(self_, scalarType, false, false, optional_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_byte(PyObject* self, PyObject* args, PyObject* kwargs)  {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "byte(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Byte, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_char(PyObject* self, PyObject* args, PyObject* kwargs)  {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "char(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Char, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_double(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "double(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Double, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_float(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "float(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Float, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_half(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "half(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Half, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_int(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "int(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Int, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_long(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "long(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Long, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_short(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "short(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Short, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_bool(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bool(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Bool, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_bfloat16(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bfloat16(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::BFloat16, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_element_size(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return THPUtils_packInt64(self_.element_size());
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc PyObjects not declarable in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_numpy(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("Converting a tensor to a NumPy array", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return torch::utils::tensor_to_numpy(self_);
  END_HANDLE_TH_ERRORS
}

// TODO: move this to ATen. We would need to expose Stream objects in ATen.
static PyObject * THPVariable_record_stream(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
#ifdef USE_CUDA
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (!THCPStream_Check(arg)) {
    return PyErr_Format(PyExc_TypeError, "expected Stream object");
  }
  c10::cuda::CUDACachingAllocator::recordStream(self_.storage().data_ptr(), at::cuda::CUDAStream::unpack(((THCPStream*)arg)->cdata));
  Py_RETURN_NONE;
#else
  throw std::runtime_error("PyTorch compiled without CUDA support");
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_requires_grad_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "requires_grad_(bool requires_grad=True)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto requires_grad = r.toBool(0);
  // should we throw if requires_grad is true?  var.requires_grad = True throws here
  // but it's nice to let this be a no-op.
  if (!self_.is_leaf() && !requires_grad) {
    throw std::runtime_error(autograd::utils::requires_grad_leaf_error(requires_grad));
  }
  if (requires_grad && !self_.is_floating_point()) {
    throw std::runtime_error("only Tensors of floating point dtype can require gradients");
  }
  self_.set_requires_grad(requires_grad);
  return THPVariable_Wrap(self_);
  END_HANDLE_TH_ERRORS
}

inline bool dispatch_is_contiguous(Tensor & self, MemoryFormat memory_format) {
  return self.is_contiguous(memory_format);
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_is_contiguous(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_contiguous(*, MemoryFormat memory_format=contiguous_format)",
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto memory_format = r.memoryformat(0);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_is_contiguous(self, memory_format));
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_item(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("Converting a tensor to a Python number", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.is_floating_point()) {
    return wrap(dispatch_to_CDouble(self_));
  } else if (self_.is_complex()) {
    return wrap(dispatch_to_CComplexDouble(self_));
  } else if (self_.scalar_type() == ScalarType::Bool) {
    return wrap(dispatch_to_Bool(self_));
  } else {
    return wrap(dispatch_to_CLong(self_));
  }
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc no support for first class functions in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_map_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({ "map_(Tensor other, PyObject* callable)" });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  Variable other = r.tensor(0);
  if (self_.requires_grad() || other.requires_grad()) {
    throw std::runtime_error(
        "Can't call map_() on Variable that requires grad. Use "
        "var.detach().map_() instead.");
  }
  return THPVariable_Wrap(torch::utils::map_(self_, other, r.pyobject(1)));
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc no support for first class functions in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_map2_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({ "map2_(Tensor x, Tensor y, PyObject* callable)" });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  Variable x = r.tensor(0);
  Variable y = r.tensor(1);
  if (self_.requires_grad() || x.requires_grad() || y.requires_grad()) {
    throw std::runtime_error(
        "Can't call map2_() on Variable that requires grad. Use "
        "var.detach().map2_() instead.");
  }
  return THPVariable_Wrap(torch::utils::map2_(self_, x, y, r.pyobject(2)));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  OptionalDeviceGuard device_guard(device_of(self_));
  return THPVariable_Wrap(torch::utils::legacy_tensor_new(legacyExtractDispatchKey(self_), self_.scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_ones(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  OptionalDeviceGuard device_guard(device_of(self_));
  return THPVariable_Wrap(torch::utils::new_ones(legacyExtractDispatchKey(self_), self_.scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  OptionalDeviceGuard device_guard(device_of(self_));
  return THPVariable_Wrap(torch::utils::new_tensor(legacyExtractDispatchKey(self_), self_.scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_storage(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return createPyObject(self_.storage());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_storage_type(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  auto storage = THPObjectPtr(createPyObject(self_.storage()));
  auto storage_type = (PyObject*)Py_TYPE(storage);
  Py_INCREF(storage_type);
  return storage_type;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_to(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto parsed = parse_to_conversion(args, kwargs, /*allow_copy*/ true);
  auto& device = std::get<0>(parsed);
  auto& scalarType = std::get<1>(parsed);
  auto non_blocking = std::get<2>(parsed);
  auto copy = std::get<3>(parsed);
  auto opt_memory_format = std::get<4>(parsed);
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (device && device->is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
  if (!device && !scalarType && !copy && !opt_memory_format.has_value()) {
    Py_INCREF(self);
    return self;
  } else if (!device && !scalarType) {
    return THPVariable_Wrap(
        dispatch_to(self_, non_blocking, copy, opt_memory_format));
  } else if (!device) {
    return THPVariable_Wrap(dispatch_to(self_, *scalarType, non_blocking, copy, opt_memory_format));
  } else if (!scalarType) {
    return THPVariable_Wrap(dispatch_to(self_, *device, non_blocking, copy, opt_memory_format));
  } else {
    return THPVariable_Wrap(dispatch_to(self_, *device, *scalarType, non_blocking, copy, opt_memory_format));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// implemented on the python object b/c arbitrarily nested list not declarable in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_tolist(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("Converting a tensor to a Python list", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return torch::utils::tensor_to_list(self_);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_type(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "type(PyObject* dtype=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "type(PyObject* dtype=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.isNone(0)) {
    return THPUtils_packString(torch::utils::options_to_string(self_.options()));
  }
  auto obj = r.pyobject(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  std::string type_name;
  bool is_dtype = false;
  if (PyType_Check(obj)) {
    if (obj == THPVariableClass) {
      type_name = "torch.Tensor";
    } else {
      type_name = ((PyTypeObject*)obj)->tp_name;
    }
  } else if (THPUtils_checkString(obj)) {
    type_name = THPUtils_unpackString(obj);
  } else if (THPDtype_Check(obj)) {
    is_dtype = true;
  } else {
    throw TypeError("dtype must be a type, str, or dtype object");
  }
  ScalarType scalar_type;
  Device device = self_.device();
  if (is_dtype) {
    scalar_type = r.scalartype(0);
  } else {
    at::TensorOptions options = torch::utils::options_from_string(type_name);
    scalar_type = at::typeMetaToScalarType(options.dtype());
    auto device_type = options.device().type();
    if (device_type != device.type()) {
      device = at::Device(device_type);
    }
  }
  if (device.is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
  return THPVariable_Wrap(dispatch_to(self_, device, scalar_type, /*non_blocking=*/ r.toBool(1), /*copy=*/ false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

// generated methods start here

\
// __and__
static PyObject * THPVariable___and__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "__and__(Tensor other)",
    "__and__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch___and__ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__and__(other);
      };
      return wrap(dispatch___and__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch___and__ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__and__(other);
      };
      return wrap(dispatch___and__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __iand__
static PyObject * THPVariable___iand__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "__iand__(Tensor other)",
    "__iand__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::__iand__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch___iand__ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__iand__(other);
      };
      return wrap(dispatch___iand__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__iand__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch___iand__ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__iand__(other);
      };
      return wrap(dispatch___iand__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __ilshift__
static PyObject * THPVariable___ilshift__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "__ilshift__(Tensor other)",
    "__ilshift__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::__ilshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch___ilshift__ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__ilshift__(other);
      };
      return wrap(dispatch___ilshift__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch___ilshift__ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__ilshift__(other);
      };
      return wrap(dispatch___ilshift__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __ior__
static PyObject * THPVariable___ior__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "__ior__(Tensor other)",
    "__ior__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::__ior__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch___ior__ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__ior__(other);
      };
      return wrap(dispatch___ior__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__ior__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch___ior__ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__ior__(other);
      };
      return wrap(dispatch___ior__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __irshift__
static PyObject * THPVariable___irshift__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "__irshift__(Tensor other)",
    "__irshift__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::__irshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch___irshift__ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__irshift__(other);
      };
      return wrap(dispatch___irshift__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__irshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch___irshift__ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__irshift__(other);
      };
      return wrap(dispatch___irshift__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __ixor__
static PyObject * THPVariable___ixor__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "__ixor__(Tensor other)",
    "__ixor__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::__ixor__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch___ixor__ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__ixor__(other);
      };
      return wrap(dispatch___ixor__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__ixor__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch___ixor__ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__ixor__(other);
      };
      return wrap(dispatch___ixor__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __lshift__
static PyObject * THPVariable___lshift__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "__lshift__(Tensor other)",
    "__lshift__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch___lshift__ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__lshift__(other);
      };
      return wrap(dispatch___lshift__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch___lshift__ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__lshift__(other);
      };
      return wrap(dispatch___lshift__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __or__
static PyObject * THPVariable___or__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "__or__(Tensor other)",
    "__or__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch___or__ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__or__(other);
      };
      return wrap(dispatch___or__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch___or__ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__or__(other);
      };
      return wrap(dispatch___or__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __rshift__
static PyObject * THPVariable___rshift__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "__rshift__(Tensor other)",
    "__rshift__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch___rshift__ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__rshift__(other);
      };
      return wrap(dispatch___rshift__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch___rshift__ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__rshift__(other);
      };
      return wrap(dispatch___rshift__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __xor__
static PyObject * THPVariable___xor__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "__xor__(Tensor other)",
    "__xor__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch___xor__ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__xor__(other);
      };
      return wrap(dispatch___xor__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch___xor__ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__xor__(other);
      };
      return wrap(dispatch___xor__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _coalesced_
static PyObject * THPVariable__coalesced_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "_coalesced_(bool coalesced)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::_coalesced_(Tensor(a!) self, bool coalesced) -> Tensor(a!)
  auto dispatch__coalesced_ = [](Tensor & self, bool coalesced) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._coalesced_(coalesced);
  };
  return wrap(dispatch__coalesced_(self, _r.toBool(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _dimI
static PyObject * THPVariable__dimI(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::_dimI(Tensor self) -> int
  auto dispatch__dimI = [](Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self._dimI();
  };
  return wrap(dispatch__dimI(self));
  END_HANDLE_TH_ERRORS
}

// _dimV
static PyObject * THPVariable__dimV(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::_dimV(Tensor self) -> int
  auto dispatch__dimV = [](Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self._dimV();
  };
  return wrap(dispatch__dimV(self));
  END_HANDLE_TH_ERRORS
}

// _indices
static PyObject * THPVariable__indices(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::_indices(Tensor(a) self) -> Tensor(a)
  auto dispatch__indices = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._indices();
  };
  return wrap(dispatch__indices(self));
  END_HANDLE_TH_ERRORS
}

// _nnz
static PyObject * THPVariable__nnz(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::_nnz(Tensor self) -> int
  auto dispatch__nnz = [](Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self._nnz();
  };
  return wrap(dispatch__nnz(self));
  END_HANDLE_TH_ERRORS
}

// _values
static PyObject * THPVariable__values(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::_values(Tensor(a) self) -> Tensor(a)
  auto dispatch__values = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._values();
  };
  return wrap(dispatch__values(self));
  END_HANDLE_TH_ERRORS
}

// abs
static PyObject * THPVariable_abs(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::abs(Tensor self) -> Tensor
  auto dispatch_abs = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.abs();
  };
  return wrap(dispatch_abs(self));
  END_HANDLE_TH_ERRORS
}

// abs_
static PyObject * THPVariable_abs_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::abs_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_abs_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.abs_();
  };
  return wrap(dispatch_abs_(self));
  END_HANDLE_TH_ERRORS
}

// acos
static PyObject * THPVariable_acos(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::acos(Tensor self) -> Tensor
  auto dispatch_acos = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.acos();
  };
  return wrap(dispatch_acos(self));
  END_HANDLE_TH_ERRORS
}

// acos_
static PyObject * THPVariable_acos_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::acos_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_acos_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.acos_();
  };
  return wrap(dispatch_acos_(self));
  END_HANDLE_TH_ERRORS
}

\
// add
static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "add(Scalar alpha, Tensor other)|deprecated",
    "add(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      auto dispatch_add = [](Tensor & self, Scalar alpha, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add(other, alpha);
      };
      return wrap(dispatch_add(self, _r.scalar(0), _r.tensor(1)));
    }
    case 1: {
      // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      auto dispatch_add = [](Tensor & self, const Tensor & other, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add(other, alpha);
      };
      return wrap(dispatch_add(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// add_
static PyObject * THPVariable_add_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "add_(Scalar alpha, Tensor other)|deprecated",
    "add_(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_add_ = [](Tensor & self, Scalar alpha, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add_(other, alpha);
      };
      return wrap(dispatch_add_(self, _r.scalar(0), _r.tensor(1)));
    }
    case 1: {
      // aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_add_ = [](Tensor & self, const Tensor & other, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add_(other, alpha);
      };
      return wrap(dispatch_add_(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addbmm
static PyObject * THPVariable_addbmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "addbmm(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_addbmm = [](Scalar beta, Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addbmm(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_addbmm(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_addbmm = [](Scalar beta, Tensor & self, const Tensor & batch1, const Tensor & batch2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addbmm(batch1, batch2, beta, 1);
      };
      return wrap(dispatch_addbmm(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_addbmm = [](Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addbmm(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_addbmm(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addbmm_
static PyObject * THPVariable_addbmm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "addbmm_(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm_(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm_(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addbmm_ = [](Scalar beta, Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addbmm_(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_addbmm_(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addbmm_ = [](Scalar beta, Tensor & self, const Tensor & batch1, const Tensor & batch2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addbmm_(batch1, batch2, beta, 1);
      };
      return wrap(dispatch_addbmm_(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addbmm_ = [](Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addbmm_(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_addbmm_(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addcdiv
static PyObject * THPVariable_addcdiv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "addcdiv(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcdiv(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
      auto dispatch_addcdiv = [](Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcdiv(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcdiv(self, _r.scalar(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
      auto dispatch_addcdiv = [](Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcdiv(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcdiv(self, _r.tensor(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addcdiv_
static PyObject * THPVariable_addcdiv_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "addcdiv_(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcdiv_(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
      auto dispatch_addcdiv_ = [](Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcdiv_(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcdiv_(self, _r.scalar(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
      auto dispatch_addcdiv_ = [](Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcdiv_(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcdiv_(self, _r.tensor(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addcmul
static PyObject * THPVariable_addcmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "addcmul(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcmul(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
      auto dispatch_addcmul = [](Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcmul(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcmul(self, _r.scalar(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
      auto dispatch_addcmul = [](Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcmul(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcmul(self, _r.tensor(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addcmul_
static PyObject * THPVariable_addcmul_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "addcmul_(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcmul_(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
      auto dispatch_addcmul_ = [](Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcmul_(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcmul_(self, _r.scalar(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
      auto dispatch_addcmul_ = [](Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcmul_(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcmul_(self, _r.tensor(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addmm
static PyObject * THPVariable_addmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "addmm(Scalar beta, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "addmm(Scalar beta, Tensor mat1, Tensor mat2)|deprecated",
    "addmm(Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_addmm = [](Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmm(mat1, mat2, beta, alpha);
      };
      return wrap(dispatch_addmm(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_addmm = [](Scalar beta, Tensor & self, const Tensor & mat1, const Tensor & mat2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmm(mat1, mat2, beta, 1);
      };
      return wrap(dispatch_addmm(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_addmm = [](Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmm(mat1, mat2, beta, alpha);
      };
      return wrap(dispatch_addmm(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addmm_
static PyObject * THPVariable_addmm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "addmm_(Scalar beta, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "addmm_(Scalar beta, Tensor mat1, Tensor mat2)|deprecated",
    "addmm_(Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addmm_ = [](Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmm_(mat1, mat2, beta, alpha);
      };
      return wrap(dispatch_addmm_(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addmm_ = [](Scalar beta, Tensor & self, const Tensor & mat1, const Tensor & mat2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmm_(mat1, mat2, beta, 1);
      };
      return wrap(dispatch_addmm_(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addmm_ = [](Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmm_(mat1, mat2, beta, alpha);
      };
      return wrap(dispatch_addmm_(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addmv
static PyObject * THPVariable_addmv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "addmv(Scalar beta, Scalar alpha, Tensor mat, Tensor vec)|deprecated",
    "addmv(Scalar beta, Tensor mat, Tensor vec)|deprecated",
    "addmv(Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_addmv = [](Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv(mat, vec, beta, alpha);
      };
      return wrap(dispatch_addmv(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_addmv = [](Scalar beta, Tensor & self, const Tensor & mat, const Tensor & vec) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv(mat, vec, beta, 1);
      };
      return wrap(dispatch_addmv(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_addmv = [](Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv(mat, vec, beta, alpha);
      };
      return wrap(dispatch_addmv(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addmv_
static PyObject * THPVariable_addmv_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "addmv_(Scalar beta, Scalar alpha, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Scalar beta, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addmv_ = [](Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv_(mat, vec, beta, alpha);
      };
      return wrap(dispatch_addmv_(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addmv_ = [](Scalar beta, Tensor & self, const Tensor & mat, const Tensor & vec) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv_(mat, vec, beta, 1);
      };
      return wrap(dispatch_addmv_(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addmv_ = [](Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv_(mat, vec, beta, alpha);
      };
      return wrap(dispatch_addmv_(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addr
static PyObject * THPVariable_addr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "addr(Scalar beta, Scalar alpha, Tensor vec1, Tensor vec2)|deprecated",
    "addr(Scalar beta, Tensor vec1, Tensor vec2)|deprecated",
    "addr(Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_addr = [](Scalar beta, Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addr(vec1, vec2, beta, alpha);
      };
      return wrap(dispatch_addr(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_addr = [](Scalar beta, Tensor & self, const Tensor & vec1, const Tensor & vec2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addr(vec1, vec2, beta, 1);
      };
      return wrap(dispatch_addr(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_addr = [](Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addr(vec1, vec2, beta, alpha);
      };
      return wrap(dispatch_addr(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addr_
static PyObject * THPVariable_addr_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "addr_(Scalar beta, Scalar alpha, Tensor vec1, Tensor vec2)|deprecated",
    "addr_(Scalar beta, Tensor vec1, Tensor vec2)|deprecated",
    "addr_(Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addr_ = [](Scalar beta, Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addr_(vec1, vec2, beta, alpha);
      };
      return wrap(dispatch_addr_(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addr_ = [](Scalar beta, Tensor & self, const Tensor & vec1, const Tensor & vec2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addr_(vec1, vec2, beta, 1);
      };
      return wrap(dispatch_addr_(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addr_ = [](Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addr_(vec1, vec2, beta, alpha);
      };
      return wrap(dispatch_addr_(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// align_as
static PyObject * THPVariable_align_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "align_as(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::align_as(Tensor self, Tensor other) -> Tensor
  auto dispatch_align_as = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.align_as(other);
  };
  return wrap(dispatch_align_as(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// align_to
static PyObject * THPVariable_align_to(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "align_to(DimnameList names)",
    "align_to(DimnameList order, int64_t ellipsis_idx)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::align_to(Tensor(a) self, Dimname[] names) -> Tensor(a)
      auto dispatch_align_to = [](Tensor & self, DimnameList names) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.align_to(names);
      };
      return wrap(dispatch_align_to(self, _r.dimnamelist(0)));
    }
    case 1: {
      // aten::align_to.ellipsis_idx(Tensor(a) self, Dimname[] order, int ellipsis_idx) -> Tensor(a)
      auto dispatch_align_to = [](Tensor & self, DimnameList order, int64_t ellipsis_idx) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.align_to(order, ellipsis_idx);
      };
      return wrap(dispatch_align_to(self, _r.dimnamelist(0), _r.toInt64(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// all
static PyObject * THPVariable_all(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "all()",
    "all(Dimname dim, bool keepdim=False)",
    "all(int64_t dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::all(Tensor self) -> Tensor
      auto dispatch_all = [](Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.all();
      };
      return wrap(dispatch_all(self));
    }
    case 1: {
      // aten::all.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
      auto dispatch_all = [](Tensor & self, Dimname dim, bool keepdim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.all(dim, keepdim);
      };
      return wrap(dispatch_all(self, _r.dimname(0), _r.toBool(1)));
    }
    case 2: {
      // aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
      auto dispatch_all = [](Tensor & self, int64_t dim, bool keepdim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.all(dim, keepdim);
      };
      return wrap(dispatch_all(self, _r.toInt64(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// allclose
static PyObject * THPVariable_allclose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "allclose(Tensor other, double rtol=1e-05, double atol=1e-08, bool equal_nan=False)",
  }, /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool
  auto dispatch_allclose = [](Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.allclose(other, rtol, atol, equal_nan);
  };
  return wrap(dispatch_allclose(self, _r.tensor(0), _r.toDouble(1), _r.toDouble(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// angle
static PyObject * THPVariable_angle(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::angle(Tensor self) -> Tensor
  auto dispatch_angle = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.angle();
  };
  return wrap(dispatch_angle(self));
  END_HANDLE_TH_ERRORS
}

\
// any
static PyObject * THPVariable_any(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "any()",
    "any(Dimname dim, bool keepdim=False)",
    "any(int64_t dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::any(Tensor self) -> Tensor
      auto dispatch_any = [](Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.any();
      };
      return wrap(dispatch_any(self));
    }
    case 1: {
      // aten::any.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
      auto dispatch_any = [](Tensor & self, Dimname dim, bool keepdim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.any(dim, keepdim);
      };
      return wrap(dispatch_any(self, _r.dimname(0), _r.toBool(1)));
    }
    case 2: {
      // aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
      auto dispatch_any = [](Tensor & self, int64_t dim, bool keepdim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.any(dim, keepdim);
      };
      return wrap(dispatch_any(self, _r.toInt64(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// argmax
static PyObject * THPVariable_argmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "argmax(int64_t? dim=None, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
  auto dispatch_argmax = [](Tensor & self, c10::optional<int64_t> dim, bool keepdim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.argmax(dim, keepdim);
  };
  return wrap(dispatch_argmax(self, _r.toInt64Optional(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// argmin
static PyObject * THPVariable_argmin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "argmin(int64_t? dim=None, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
  auto dispatch_argmin = [](Tensor & self, c10::optional<int64_t> dim, bool keepdim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.argmin(dim, keepdim);
  };
  return wrap(dispatch_argmin(self, _r.toInt64Optional(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// argsort
static PyObject * THPVariable_argsort(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "argsort(Dimname dim, bool descending=False)",
    "argsort(int64_t dim=-1, bool descending=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::argsort.dimname(Tensor self, Dimname dim, bool descending=False) -> Tensor
      auto dispatch_argsort = [](Tensor & self, Dimname dim, bool descending) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.argsort(dim, descending);
      };
      return wrap(dispatch_argsort(self, _r.dimname(0), _r.toBool(1)));
    }
    case 1: {
      // aten::argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor
      auto dispatch_argsort = [](Tensor & self, int64_t dim, bool descending) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.argsort(dim, descending);
      };
      return wrap(dispatch_argsort(self, _r.toInt64(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// as_strided
static PyObject * THPVariable_as_strided(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "as_strided(IntArrayRef size, IntArrayRef stride, int64_t? storage_offset=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)
  auto dispatch_as_strided = [](Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.as_strided(size, stride, storage_offset);
  };
  return wrap(dispatch_as_strided(self, _r.intlist(0), _r.intlist(1), _r.toInt64Optional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// as_strided_
static PyObject * THPVariable_as_strided_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "as_strided_(IntArrayRef size, IntArrayRef stride, int64_t? storage_offset=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::as_strided_(Tensor(a!) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a!)
  auto dispatch_as_strided_ = [](Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.as_strided_(size, stride, storage_offset);
  };
  return wrap(dispatch_as_strided_(self, _r.intlist(0), _r.intlist(1), _r.toInt64Optional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// asin
static PyObject * THPVariable_asin(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::asin(Tensor self) -> Tensor
  auto dispatch_asin = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.asin();
  };
  return wrap(dispatch_asin(self));
  END_HANDLE_TH_ERRORS
}

// asin_
static PyObject * THPVariable_asin_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::asin_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_asin_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.asin_();
  };
  return wrap(dispatch_asin_(self));
  END_HANDLE_TH_ERRORS
}

// atan
static PyObject * THPVariable_atan(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::atan(Tensor self) -> Tensor
  auto dispatch_atan = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.atan();
  };
  return wrap(dispatch_atan(self));
  END_HANDLE_TH_ERRORS
}

// atan2
static PyObject * THPVariable_atan2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "atan2(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::atan2(Tensor self, Tensor other) -> Tensor
  auto dispatch_atan2 = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.atan2(other);
  };
  return wrap(dispatch_atan2(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// atan2_
static PyObject * THPVariable_atan2_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "atan2_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  auto dispatch_atan2_ = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.atan2_(other);
  };
  return wrap(dispatch_atan2_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// atan_
static PyObject * THPVariable_atan_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::atan_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_atan_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.atan_();
  };
  return wrap(dispatch_atan_(self));
  END_HANDLE_TH_ERRORS
}

// backward
static PyObject * THPVariable_backward(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "backward(Tensor? gradient=None, bool keep_graph=False, bool create_graph=False)",
  }, /*traceable=*/false);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::backward(Tensor self, Tensor? gradient=None, bool keep_graph=False, bool create_graph=False) -> ()
  auto dispatch_backward = [](Tensor & self, const Tensor & gradient, bool keep_graph, bool create_graph) -> void {
    pybind11::gil_scoped_release no_gil;
    self.backward(gradient, keep_graph, create_graph);
  };
  dispatch_backward(self, _r.tensor(0), _r.toBool(1), _r.toBool(2));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// baddbmm
static PyObject * THPVariable_baddbmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "baddbmm(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_baddbmm = [](Scalar beta, Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.baddbmm(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_baddbmm(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_baddbmm = [](Scalar beta, Tensor & self, const Tensor & batch1, const Tensor & batch2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.baddbmm(batch1, batch2, beta, 1);
      };
      return wrap(dispatch_baddbmm(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_baddbmm = [](Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.baddbmm(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_baddbmm(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// baddbmm_
static PyObject * THPVariable_baddbmm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "baddbmm_(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm_(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm_(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_baddbmm_ = [](Scalar beta, Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.baddbmm_(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_baddbmm_(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_baddbmm_ = [](Scalar beta, Tensor & self, const Tensor & batch1, const Tensor & batch2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.baddbmm_(batch1, batch2, beta, 1);
      };
      return wrap(dispatch_baddbmm_(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_baddbmm_ = [](Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.baddbmm_(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_baddbmm_(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bernoulli
static PyObject * THPVariable_bernoulli(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "bernoulli(*, Generator generator=None)",
    "bernoulli(double p, *, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor
      auto dispatch_bernoulli = [](Tensor & self, Generator * generator) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bernoulli(generator);
      };
      return wrap(dispatch_bernoulli(self, _r.generator(0)));
    }
    case 1: {
      // aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor
      auto dispatch_bernoulli = [](Tensor & self, double p, Generator * generator) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bernoulli(p, generator);
      };
      return wrap(dispatch_bernoulli(self, _r.toDouble(0), _r.generator(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bernoulli_
static PyObject * THPVariable_bernoulli_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "bernoulli_(Tensor p, *, Generator generator=None)",
    "bernoulli_(double p=0.5, *, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)
      auto dispatch_bernoulli_ = [](Tensor & self, const Tensor & p, Generator * generator) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bernoulli_(p, generator);
      };
      return wrap(dispatch_bernoulli_(self, _r.tensor(0), _r.generator(1)));
    }
    case 1: {
      // aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)
      auto dispatch_bernoulli_ = [](Tensor & self, double p, Generator * generator) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bernoulli_(p, generator);
      };
      return wrap(dispatch_bernoulli_(self, _r.toDouble(0), _r.generator(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// bincount
static PyObject * THPVariable_bincount(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "bincount(Tensor? weights=None, int64_t minlength=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor
  auto dispatch_bincount = [](Tensor & self, const Tensor & weights, int64_t minlength) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.bincount(weights, minlength);
  };
  return wrap(dispatch_bincount(self, _r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_and
static PyObject * THPVariable_bitwise_and(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "bitwise_and(Tensor other)",
    "bitwise_and(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch_bitwise_and = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_and(other);
      };
      return wrap(dispatch_bitwise_and(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch_bitwise_and = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_and(other);
      };
      return wrap(dispatch_bitwise_and(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_and_
static PyObject * THPVariable_bitwise_and_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "bitwise_and_(Tensor other)",
    "bitwise_and_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::bitwise_and_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch_bitwise_and_ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_and_(other);
      };
      return wrap(dispatch_bitwise_and_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_and_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch_bitwise_and_ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_and_(other);
      };
      return wrap(dispatch_bitwise_and_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// bitwise_not
static PyObject * THPVariable_bitwise_not(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::bitwise_not(Tensor self) -> Tensor
  auto dispatch_bitwise_not = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.bitwise_not();
  };
  return wrap(dispatch_bitwise_not(self));
  END_HANDLE_TH_ERRORS
}

// bitwise_not_
static PyObject * THPVariable_bitwise_not_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_bitwise_not_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.bitwise_not_();
  };
  return wrap(dispatch_bitwise_not_(self));
  END_HANDLE_TH_ERRORS
}

\
// bitwise_or
static PyObject * THPVariable_bitwise_or(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "bitwise_or(Tensor other)",
    "bitwise_or(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch_bitwise_or = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_or(other);
      };
      return wrap(dispatch_bitwise_or(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch_bitwise_or = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_or(other);
      };
      return wrap(dispatch_bitwise_or(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_or_
static PyObject * THPVariable_bitwise_or_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "bitwise_or_(Tensor other)",
    "bitwise_or_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::bitwise_or_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch_bitwise_or_ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_or_(other);
      };
      return wrap(dispatch_bitwise_or_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_or_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch_bitwise_or_ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_or_(other);
      };
      return wrap(dispatch_bitwise_or_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_xor
static PyObject * THPVariable_bitwise_xor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "bitwise_xor(Tensor other)",
    "bitwise_xor(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch_bitwise_xor = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_xor(other);
      };
      return wrap(dispatch_bitwise_xor(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch_bitwise_xor = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_xor(other);
      };
      return wrap(dispatch_bitwise_xor(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_xor_
static PyObject * THPVariable_bitwise_xor_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "bitwise_xor_(Tensor other)",
    "bitwise_xor_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::bitwise_xor_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch_bitwise_xor_ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_xor_(other);
      };
      return wrap(dispatch_bitwise_xor_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_xor_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch_bitwise_xor_ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_xor_(other);
      };
      return wrap(dispatch_bitwise_xor_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// bmm
static PyObject * THPVariable_bmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "bmm(Tensor mat2)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::bmm(Tensor self, Tensor mat2) -> Tensor
  auto dispatch_bmm = [](Tensor & self, const Tensor & mat2) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.bmm(mat2);
  };
  return wrap(dispatch_bmm(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cauchy_
static PyObject * THPVariable_cauchy_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "cauchy_(double median=0, double sigma=1, *, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)
  auto dispatch_cauchy_ = [](Tensor & self, double median, double sigma, Generator * generator) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cauchy_(median, sigma, generator);
  };
  return wrap(dispatch_cauchy_(self, _r.toDouble(0), _r.toDouble(1), _r.generator(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ceil
static PyObject * THPVariable_ceil(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::ceil(Tensor self) -> Tensor
  auto dispatch_ceil = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ceil();
  };
  return wrap(dispatch_ceil(self));
  END_HANDLE_TH_ERRORS
}

// ceil_
static PyObject * THPVariable_ceil_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::ceil_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_ceil_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ceil_();
  };
  return wrap(dispatch_ceil_(self));
  END_HANDLE_TH_ERRORS
}

// cholesky
static PyObject * THPVariable_cholesky(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "cholesky(bool upper=False)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::cholesky(Tensor self, bool upper=False) -> Tensor
  auto dispatch_cholesky = [](Tensor & self, bool upper) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cholesky(upper);
  };
  return wrap(dispatch_cholesky(self, _r.toBool(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cholesky_inverse
static PyObject * THPVariable_cholesky_inverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "cholesky_inverse(bool upper=False)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor
  auto dispatch_cholesky_inverse = [](Tensor & self, bool upper) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cholesky_inverse(upper);
  };
  return wrap(dispatch_cholesky_inverse(self, _r.toBool(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cholesky_solve
static PyObject * THPVariable_cholesky_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "cholesky_solve(Tensor input2, bool upper=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor
  auto dispatch_cholesky_solve = [](Tensor & self, const Tensor & input2, bool upper) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cholesky_solve(input2, upper);
  };
  return wrap(dispatch_cholesky_solve(self, _r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// chunk
static PyObject * THPVariable_chunk(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "chunk(int64_t chunks, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]
  auto dispatch_chunk = [](Tensor & self, int64_t chunks, int64_t dim) -> std::vector<Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.chunk(chunks, dim);
  };
  return wrap(dispatch_chunk(self, _r.toInt64(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clamp
static PyObject * THPVariable_clamp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "clamp(Scalar? min=None, Scalar? max=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
  auto dispatch_clamp = [](Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.clamp(min, max);
  };
  return wrap(dispatch_clamp(self, _r.scalarOptional(0), _r.scalarOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clamp_
static PyObject * THPVariable_clamp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "clamp_(Scalar? min=None, Scalar? max=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
  auto dispatch_clamp_ = [](Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.clamp_(min, max);
  };
  return wrap(dispatch_clamp_(self, _r.scalarOptional(0), _r.scalarOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clamp_max
static PyObject * THPVariable_clamp_max(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "clamp_max(Scalar max)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::clamp_max(Tensor self, Scalar max) -> Tensor
  auto dispatch_clamp_max = [](Tensor & self, Scalar max) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.clamp_max(max);
  };
  return wrap(dispatch_clamp_max(self, _r.scalar(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clamp_max_
static PyObject * THPVariable_clamp_max_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "clamp_max_(Scalar max)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)
  auto dispatch_clamp_max_ = [](Tensor & self, Scalar max) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.clamp_max_(max);
  };
  return wrap(dispatch_clamp_max_(self, _r.scalar(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clamp_min
static PyObject * THPVariable_clamp_min(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "clamp_min(Scalar min)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::clamp_min(Tensor self, Scalar min) -> Tensor
  auto dispatch_clamp_min = [](Tensor & self, Scalar min) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.clamp_min(min);
  };
  return wrap(dispatch_clamp_min(self, _r.scalar(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clamp_min_
static PyObject * THPVariable_clamp_min_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "clamp_min_(Scalar min)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)
  auto dispatch_clamp_min_ = [](Tensor & self, Scalar min) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.clamp_min_(min);
  };
  return wrap(dispatch_clamp_min_(self, _r.scalar(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clone
static PyObject * THPVariable_clone(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "clone(*, MemoryFormat? memory_format=None)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
  auto dispatch_clone = [](Tensor & self, c10::optional<MemoryFormat> memory_format) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.clone(memory_format);
  };
  return wrap(dispatch_clone(self, _r.memoryformatOptional(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// coalesce
static PyObject * THPVariable_coalesce(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::coalesce(Tensor self) -> Tensor
  auto dispatch_coalesce = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.coalesce();
  };
  return wrap(dispatch_coalesce(self));
  END_HANDLE_TH_ERRORS
}

// conj
static PyObject * THPVariable_conj(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::conj(Tensor self) -> Tensor
  auto dispatch_conj = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.conj();
  };
  return wrap(dispatch_conj(self));
  END_HANDLE_TH_ERRORS
}

// cos
static PyObject * THPVariable_cos(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::cos(Tensor self) -> Tensor
  auto dispatch_cos = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cos();
  };
  return wrap(dispatch_cos(self));
  END_HANDLE_TH_ERRORS
}

// cos_
static PyObject * THPVariable_cos_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::cos_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_cos_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cos_();
  };
  return wrap(dispatch_cos_(self));
  END_HANDLE_TH_ERRORS
}

// cosh
static PyObject * THPVariable_cosh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::cosh(Tensor self) -> Tensor
  auto dispatch_cosh = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cosh();
  };
  return wrap(dispatch_cosh(self));
  END_HANDLE_TH_ERRORS
}

// cosh_
static PyObject * THPVariable_cosh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::cosh_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_cosh_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cosh_();
  };
  return wrap(dispatch_cosh_(self));
  END_HANDLE_TH_ERRORS
}

// cross
static PyObject * THPVariable_cross(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "cross(Tensor other, int64_t? dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor
  auto dispatch_cross = [](Tensor & self, const Tensor & other, c10::optional<int64_t> dim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cross(other, dim);
  };
  return wrap(dispatch_cross(self, _r.tensor(0), _r.toInt64Optional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cummax
static PyObject * THPVariable_cummax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.cummax", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "cummax(Dimname dim)",
    "cummax(int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::cummax.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
      auto dispatch_cummax = [](Tensor & self, Dimname dim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.cummax(dim);
      };
      return wrap(&NamedTuple, dispatch_cummax(self, _r.dimname(0)));
    }
    case 1: {
      // aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
      auto dispatch_cummax = [](Tensor & self, int64_t dim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.cummax(dim);
      };
      return wrap(&NamedTuple, dispatch_cummax(self, _r.toInt64(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cummin
static PyObject * THPVariable_cummin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.cummin", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "cummin(Dimname dim)",
    "cummin(int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::cummin.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
      auto dispatch_cummin = [](Tensor & self, Dimname dim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.cummin(dim);
      };
      return wrap(&NamedTuple, dispatch_cummin(self, _r.dimname(0)));
    }
    case 1: {
      // aten::cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)
      auto dispatch_cummin = [](Tensor & self, int64_t dim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.cummin(dim);
      };
      return wrap(&NamedTuple, dispatch_cummin(self, _r.toInt64(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cumprod
static PyObject * THPVariable_cumprod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "cumprod(Dimname dim, *, ScalarType? dtype=None)",
    "cumprod(int64_t dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::cumprod.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_cumprod = [](Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.cumprod(dim, dtype);
      };
      return wrap(dispatch_cumprod(self, _r.dimname(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      // aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_cumprod = [](Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.cumprod(dim, dtype);
      };
      return wrap(dispatch_cumprod(self, _r.toInt64(0), _r.scalartypeOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cumsum
static PyObject * THPVariable_cumsum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "cumsum(Dimname dim, *, ScalarType? dtype=None)",
    "cumsum(int64_t dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::cumsum.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_cumsum = [](Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.cumsum(dim, dtype);
      };
      return wrap(dispatch_cumsum(self, _r.dimname(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      // aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_cumsum = [](Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.cumsum(dim, dtype);
      };
      return wrap(dispatch_cumsum(self, _r.toInt64(0), _r.scalartypeOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// dense_dim
static PyObject * THPVariable_dense_dim(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::dense_dim(Tensor self) -> int
  auto dispatch_dense_dim = [](Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self.dense_dim();
  };
  return wrap(dispatch_dense_dim(self));
  END_HANDLE_TH_ERRORS
}

// dequantize
static PyObject * THPVariable_dequantize(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::dequantize(Tensor self) -> Tensor
  auto dispatch_dequantize = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.dequantize();
  };
  return wrap(dispatch_dequantize(self));
  END_HANDLE_TH_ERRORS
}

// det
static PyObject * THPVariable_det(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::det(Tensor self) -> Tensor
  auto dispatch_det = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.det();
  };
  return wrap(dispatch_det(self));
  END_HANDLE_TH_ERRORS
}

// detach
static PyObject * THPVariable_detach(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::detach(Tensor self) -> Tensor
  auto dispatch_detach = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.detach();
  };
  return wrap(dispatch_detach(self));
  END_HANDLE_TH_ERRORS
}

// detach_
static PyObject * THPVariable_detach_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::detach_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_detach_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.detach_();
  };
  return wrap(dispatch_detach_(self));
  END_HANDLE_TH_ERRORS
}

// diag
static PyObject * THPVariable_diag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "diag(int64_t diagonal=0)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::diag(Tensor self, int diagonal=0) -> Tensor
  auto dispatch_diag = [](Tensor & self, int64_t diagonal) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.diag(diagonal);
  };
  return wrap(dispatch_diag(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diag_embed
static PyObject * THPVariable_diag_embed(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "diag_embed(int64_t offset=0, int64_t dim1=-2, int64_t dim2=-1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor
  auto dispatch_diag_embed = [](Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.diag_embed(offset, dim1, dim2);
  };
  return wrap(dispatch_diag_embed(self, _r.toInt64(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diagflat
static PyObject * THPVariable_diagflat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "diagflat(int64_t offset=0)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::diagflat(Tensor self, int offset=0) -> Tensor
  auto dispatch_diagflat = [](Tensor & self, int64_t offset) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.diagflat(offset);
  };
  return wrap(dispatch_diagflat(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// diagonal
static PyObject * THPVariable_diagonal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "diagonal(*, Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset=0)",
    "diagonal(int64_t offset=0, int64_t dim1=0, int64_t dim2=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::diagonal.Dimname(Tensor(a) self, *, Dimname outdim, Dimname dim1, Dimname dim2, int offset=0) -> Tensor(a)
      auto dispatch_diagonal = [](Tensor & self, Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.diagonal(outdim, dim1, dim2, offset);
      };
      return wrap(dispatch_diagonal(self, _r.dimname(0), _r.dimname(1), _r.dimname(2), _r.toInt64(3)));
    }
    case 1: {
      // aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
      auto dispatch_diagonal = [](Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.diagonal(offset, dim1, dim2);
      };
      return wrap(dispatch_diagonal(self, _r.toInt64(0), _r.toInt64(1), _r.toInt64(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// digamma
static PyObject * THPVariable_digamma(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::digamma(Tensor self) -> Tensor
  auto dispatch_digamma = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.digamma();
  };
  return wrap(dispatch_digamma(self));
  END_HANDLE_TH_ERRORS
}

// digamma_
static PyObject * THPVariable_digamma_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::digamma_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_digamma_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.digamma_();
  };
  return wrap(dispatch_digamma_(self));
  END_HANDLE_TH_ERRORS
}

// dist
static PyObject * THPVariable_dist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "dist(Tensor other, Scalar p=2)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor
  auto dispatch_dist = [](Tensor & self, const Tensor & other, Scalar p) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.dist(other, p);
  };
  return wrap(dispatch_dist(self, _r.tensor(0), _r.scalar(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// div
static PyObject * THPVariable_div(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "div(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::div.Tensor(Tensor self, Tensor other) -> Tensor
  auto dispatch_div = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.div(other);
  };
  return wrap(dispatch_div(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// div_
static PyObject * THPVariable_div_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "div_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
  auto dispatch_div_ = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.div_(other);
  };
  return wrap(dispatch_div_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// dot
static PyObject * THPVariable_dot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "dot(Tensor tensor)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::dot(Tensor self, Tensor tensor) -> Tensor
  auto dispatch_dot = [](Tensor & self, const Tensor & tensor) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.dot(tensor);
  };
  return wrap(dispatch_dot(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// eig
static PyObject * THPVariable_eig(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"eigenvalues", ""}, {"eigenvectors", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.eig", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "eig(bool eigenvectors=False)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::eig(Tensor self, bool eigenvectors=False) -> (Tensor eigenvalues, Tensor eigenvectors)
  auto dispatch_eig = [](Tensor & self, bool eigenvectors) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.eig(eigenvectors);
  };
  return wrap(&NamedTuple, dispatch_eig(self, _r.toBool(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// eq
static PyObject * THPVariable_eq(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "eq(Tensor other)",
    "eq(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::eq.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch_eq = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.eq(other);
      };
      return wrap(dispatch_eq(self, _r.tensor(0)));
    }
    case 1: {
      // aten::eq.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch_eq = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.eq(other);
      };
      return wrap(dispatch_eq(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// eq_
static PyObject * THPVariable_eq_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "eq_(Tensor other)",
    "eq_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch_eq_ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.eq_(other);
      };
      return wrap(dispatch_eq_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch_eq_ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.eq_(other);
      };
      return wrap(dispatch_eq_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// equal
static PyObject * THPVariable_equal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "equal(Tensor other)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::equal(Tensor self, Tensor other) -> bool
  auto dispatch_equal = [](Tensor & self, const Tensor & other) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.equal(other);
  };
  return wrap(dispatch_equal(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// erf
static PyObject * THPVariable_erf(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::erf(Tensor self) -> Tensor
  auto dispatch_erf = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erf();
  };
  return wrap(dispatch_erf(self));
  END_HANDLE_TH_ERRORS
}

// erf_
static PyObject * THPVariable_erf_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::erf_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_erf_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erf_();
  };
  return wrap(dispatch_erf_(self));
  END_HANDLE_TH_ERRORS
}

// erfc
static PyObject * THPVariable_erfc(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::erfc(Tensor self) -> Tensor
  auto dispatch_erfc = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erfc();
  };
  return wrap(dispatch_erfc(self));
  END_HANDLE_TH_ERRORS
}

// erfc_
static PyObject * THPVariable_erfc_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::erfc_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_erfc_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erfc_();
  };
  return wrap(dispatch_erfc_(self));
  END_HANDLE_TH_ERRORS
}

// erfinv
static PyObject * THPVariable_erfinv(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::erfinv(Tensor self) -> Tensor
  auto dispatch_erfinv = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erfinv();
  };
  return wrap(dispatch_erfinv(self));
  END_HANDLE_TH_ERRORS
}

// erfinv_
static PyObject * THPVariable_erfinv_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::erfinv_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_erfinv_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erfinv_();
  };
  return wrap(dispatch_erfinv_(self));
  END_HANDLE_TH_ERRORS
}

// exp
static PyObject * THPVariable_exp(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::exp(Tensor self) -> Tensor
  auto dispatch_exp = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.exp();
  };
  return wrap(dispatch_exp(self));
  END_HANDLE_TH_ERRORS
}

// exp_
static PyObject * THPVariable_exp_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::exp_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_exp_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.exp_();
  };
  return wrap(dispatch_exp_(self));
  END_HANDLE_TH_ERRORS
}

// expand
static PyObject * THPVariable_expand(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "expand(IntArrayRef size, *, bool implicit=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)
  auto dispatch_expand = [](Tensor & self, IntArrayRef size, bool implicit) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.expand(size, implicit);
  };
  return wrap(dispatch_expand(self, _r.intlist(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// expand_as
static PyObject * THPVariable_expand_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "expand_as(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::expand_as(Tensor self, Tensor other) -> Tensor
  auto dispatch_expand_as = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.expand_as(other);
  };
  return wrap(dispatch_expand_as(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// expm1
static PyObject * THPVariable_expm1(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::expm1(Tensor self) -> Tensor
  auto dispatch_expm1 = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.expm1();
  };
  return wrap(dispatch_expm1(self));
  END_HANDLE_TH_ERRORS
}

// expm1_
static PyObject * THPVariable_expm1_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::expm1_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_expm1_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.expm1_();
  };
  return wrap(dispatch_expm1_(self));
  END_HANDLE_TH_ERRORS
}

// exponential_
static PyObject * THPVariable_exponential_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "exponential_(double lambd=1, *, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)
  auto dispatch_exponential_ = [](Tensor & self, double lambd, Generator * generator) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.exponential_(lambd, generator);
  };
  return wrap(dispatch_exponential_(self, _r.toDouble(0), _r.generator(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft
static PyObject * THPVariable_fft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "fft(int64_t signal_ndim, bool normalized=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::fft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor
  auto dispatch_fft = [](Tensor & self, int64_t signal_ndim, bool normalized) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.fft(signal_ndim, normalized);
  };
  return wrap(dispatch_fft(self, _r.toInt64(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// fill_
static PyObject * THPVariable_fill_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "fill_(Tensor value)",
    "fill_(Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
      auto dispatch_fill_ = [](Tensor & self, const Tensor & value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fill_(value);
      };
      return wrap(dispatch_fill_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
      auto dispatch_fill_ = [](Tensor & self, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fill_(value);
      };
      return wrap(dispatch_fill_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fill_diagonal_
static PyObject * THPVariable_fill_diagonal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "fill_diagonal_(Scalar fill_value, bool wrap=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::fill_diagonal_(Tensor(a!) self, Scalar fill_value, bool wrap=False) -> Tensor(a!)
  auto dispatch_fill_diagonal_ = [](Tensor & self, Scalar fill_value, bool wrap) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.fill_diagonal_(fill_value, wrap);
  };
  return wrap(dispatch_fill_diagonal_(self, _r.scalar(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// flatten
static PyObject * THPVariable_flatten(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "flatten(Dimname start_dim, Dimname end_dim, Dimname out_dim)",
    "flatten(DimnameList dims, Dimname out_dim)",
    "flatten(int64_t start_dim, int64_t end_dim, Dimname out_dim)",
    "flatten(int64_t start_dim=0, int64_t end_dim=-1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::flatten.using_names(Tensor self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor
      auto dispatch_flatten = [](Tensor & self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(start_dim, end_dim, out_dim);
      };
      return wrap(dispatch_flatten(self, _r.dimname(0), _r.dimname(1), _r.dimname(2)));
    }
    case 1: {
      // aten::flatten.DimnameList(Tensor self, Dimname[] dims, Dimname out_dim) -> Tensor
      auto dispatch_flatten = [](Tensor & self, DimnameList dims, Dimname out_dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(dims, out_dim);
      };
      return wrap(dispatch_flatten(self, _r.dimnamelist(0), _r.dimname(1)));
    }
    case 2: {
      // aten::flatten.named_out_dim(Tensor self, int start_dim, int end_dim, Dimname out_dim) -> Tensor
      auto dispatch_flatten = [](Tensor & self, int64_t start_dim, int64_t end_dim, Dimname out_dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(start_dim, end_dim, out_dim);
      };
      return wrap(dispatch_flatten(self, _r.toInt64(0), _r.toInt64(1), _r.dimname(2)));
    }
    case 3: {
      // aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor
      auto dispatch_flatten = [](Tensor & self, int64_t start_dim, int64_t end_dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(start_dim, end_dim);
      };
      return wrap(dispatch_flatten(self, _r.toInt64(0), _r.toInt64(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// flip
static PyObject * THPVariable_flip(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "flip(IntArrayRef dims)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::flip(Tensor self, int[] dims) -> Tensor
  auto dispatch_flip = [](Tensor & self, IntArrayRef dims) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.flip(dims);
  };
  return wrap(dispatch_flip(self, _r.intlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// floor
static PyObject * THPVariable_floor(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::floor(Tensor self) -> Tensor
  auto dispatch_floor = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.floor();
  };
  return wrap(dispatch_floor(self));
  END_HANDLE_TH_ERRORS
}

// floor_
static PyObject * THPVariable_floor_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::floor_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_floor_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.floor_();
  };
  return wrap(dispatch_floor_(self));
  END_HANDLE_TH_ERRORS
}

\
// fmod
static PyObject * THPVariable_fmod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "fmod(Tensor other)",
    "fmod(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch_fmod = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fmod(other);
      };
      return wrap(dispatch_fmod(self, _r.tensor(0)));
    }
    case 1: {
      // aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch_fmod = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fmod(other);
      };
      return wrap(dispatch_fmod(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// fmod_
static PyObject * THPVariable_fmod_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "fmod_(Tensor other)",
    "fmod_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::fmod_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch_fmod_ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fmod_(other);
      };
      return wrap(dispatch_fmod_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::fmod_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch_fmod_ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fmod_(other);
      };
      return wrap(dispatch_fmod_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// frac
static PyObject * THPVariable_frac(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::frac(Tensor self) -> Tensor
  auto dispatch_frac = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.frac();
  };
  return wrap(dispatch_frac(self));
  END_HANDLE_TH_ERRORS
}

// frac_
static PyObject * THPVariable_frac_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::frac_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_frac_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.frac_();
  };
  return wrap(dispatch_frac_(self));
  END_HANDLE_TH_ERRORS
}

\
// gather
static PyObject * THPVariable_gather(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "gather(Dimname dim, Tensor index, *, bool sparse_grad=False)",
    "gather(int64_t dim, Tensor index, *, bool sparse_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::gather.dimname(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False) -> Tensor
      auto dispatch_gather = [](Tensor & self, Dimname dim, const Tensor & index, bool sparse_grad) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.gather(dim, index, sparse_grad);
      };
      return wrap(dispatch_gather(self, _r.dimname(0), _r.tensor(1), _r.toBool(2)));
    }
    case 1: {
      // aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
      auto dispatch_gather = [](Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.gather(dim, index, sparse_grad);
      };
      return wrap(dispatch_gather(self, _r.toInt64(0), _r.tensor(1), _r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ge
static PyObject * THPVariable_ge(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "ge(Tensor other)",
    "ge(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::ge.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch_ge = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ge(other);
      };
      return wrap(dispatch_ge(self, _r.tensor(0)));
    }
    case 1: {
      // aten::ge.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch_ge = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ge(other);
      };
      return wrap(dispatch_ge(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ge_
static PyObject * THPVariable_ge_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "ge_(Tensor other)",
    "ge_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::ge_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch_ge_ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ge_(other);
      };
      return wrap(dispatch_ge_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::ge_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch_ge_ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ge_(other);
      };
      return wrap(dispatch_ge_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// geometric_
static PyObject * THPVariable_geometric_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "geometric_(double p, *, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
  auto dispatch_geometric_ = [](Tensor & self, double p, Generator * generator) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.geometric_(p, generator);
  };
  return wrap(dispatch_geometric_(self, _r.toDouble(0), _r.generator(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// geqrf
static PyObject * THPVariable_geqrf(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"a", ""}, {"tau", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.geqrf", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)
  auto dispatch_geqrf = [](Tensor & self) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.geqrf();
  };
  return wrap(&NamedTuple, dispatch_geqrf(self));
  END_HANDLE_TH_ERRORS
}

// ger
static PyObject * THPVariable_ger(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "ger(Tensor vec2)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::ger(Tensor self, Tensor vec2) -> Tensor
  auto dispatch_ger = [](Tensor & self, const Tensor & vec2) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ger(vec2);
  };
  return wrap(dispatch_ger(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// gt
static PyObject * THPVariable_gt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "gt(Tensor other)",
    "gt(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::gt.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch_gt = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.gt(other);
      };
      return wrap(dispatch_gt(self, _r.tensor(0)));
    }
    case 1: {
      // aten::gt.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch_gt = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.gt(other);
      };
      return wrap(dispatch_gt(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// gt_
static PyObject * THPVariable_gt_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "gt_(Tensor other)",
    "gt_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch_gt_ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.gt_(other);
      };
      return wrap(dispatch_gt_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch_gt_ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.gt_(other);
      };
      return wrap(dispatch_gt_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hardshrink
static PyObject * THPVariable_hardshrink(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "hardshrink(Scalar lambd=0.5)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor
  auto dispatch_hardshrink = [](Tensor & self, Scalar lambd) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.hardshrink(lambd);
  };
  return wrap(dispatch_hardshrink(self, _r.scalar(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// histc
static PyObject * THPVariable_histc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "histc(int64_t bins=100, Scalar min=0, Scalar max=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
  auto dispatch_histc = [](Tensor & self, int64_t bins, Scalar min, Scalar max) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.histc(bins, min, max);
  };
  return wrap(dispatch_histc(self, _r.toInt64(0), _r.scalar(1), _r.scalar(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ifft
static PyObject * THPVariable_ifft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "ifft(int64_t signal_ndim, bool normalized=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::ifft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor
  auto dispatch_ifft = [](Tensor & self, int64_t signal_ndim, bool normalized) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ifft(signal_ndim, normalized);
  };
  return wrap(dispatch_ifft(self, _r.toInt64(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// imag
static PyObject * THPVariable_imag(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::imag(Tensor self) -> Tensor
  auto dispatch_imag = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.imag();
  };
  return wrap(dispatch_imag(self));
  END_HANDLE_TH_ERRORS
}

\
// index_add
static PyObject * THPVariable_index_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "index_add(Dimname dim, Tensor index, Tensor source)",
    "index_add(int64_t dim, Tensor index, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::index_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor source) -> Tensor
      auto dispatch_index_add = [](Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_add(dim, index, source);
      };
      return wrap(dispatch_index_add(self, _r.dimname(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::index_add(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
      auto dispatch_index_add = [](Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_add(dim, index, source);
      };
      return wrap(dispatch_index_add(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// index_add_
static PyObject * THPVariable_index_add_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "index_add_(int64_t dim, Tensor index, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
  auto dispatch_index_add_ = [](Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.index_add_(dim, index, source);
  };
  return wrap(dispatch_index_add_(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_copy
static PyObject * THPVariable_index_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "index_copy(Dimname dim, Tensor index, Tensor source)",
    "index_copy(int64_t dim, Tensor index, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::index_copy.dimname(Tensor self, Dimname dim, Tensor index, Tensor source) -> Tensor
      auto dispatch_index_copy = [](Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_copy(dim, index, source);
      };
      return wrap(dispatch_index_copy(self, _r.dimname(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
      auto dispatch_index_copy = [](Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_copy(dim, index, source);
      };
      return wrap(dispatch_index_copy(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_copy_
static PyObject * THPVariable_index_copy_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "index_copy_(Dimname dim, Tensor index, Tensor source)",
    "index_copy_(int64_t dim, Tensor index, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::index_copy_.dimname(Tensor(a!) self, Dimname dim, Tensor index, Tensor source) -> Tensor(a!)
      auto dispatch_index_copy_ = [](Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_copy_(dim, index, source);
      };
      return wrap(dispatch_index_copy_(self, _r.dimname(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
      auto dispatch_index_copy_ = [](Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_copy_(dim, index, source);
      };
      return wrap(dispatch_index_copy_(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_fill
static PyObject * THPVariable_index_fill(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "index_fill(Dimname dim, Tensor index, Tensor value)",
    "index_fill(int64_t dim, Tensor index, Tensor value)",
    "index_fill(Dimname dim, Tensor index, Scalar value)",
    "index_fill(int64_t dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::index_fill.Dimname_Tensor(Tensor self, Dimname dim, Tensor index, Tensor value) -> Tensor
      auto dispatch_index_fill = [](Tensor & self, Dimname dim, const Tensor & index, const Tensor & value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(self, _r.dimname(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor
      auto dispatch_index_fill = [](Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::index_fill.Dimname_Scalar(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
      auto dispatch_index_fill = [](Tensor & self, Dimname dim, const Tensor & index, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(self, _r.dimname(0), _r.tensor(1), _r.scalar(2)));
    }
    case 3: {
      // aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
      auto dispatch_index_fill = [](Tensor & self, int64_t dim, const Tensor & index, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(self, _r.toInt64(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_fill_
static PyObject * THPVariable_index_fill_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "index_fill_(Dimname dim, Tensor index, Tensor value)",
    "index_fill_(int64_t dim, Tensor index, Tensor value)",
    "index_fill_(Dimname dim, Tensor index, Scalar value)",
    "index_fill_(int64_t dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::index_fill_.Dimname_Tensor(Tensor(a!) self, Dimname dim, Tensor index, Tensor value) -> Tensor(a!)
      auto dispatch_index_fill_ = [](Tensor & self, Dimname dim, const Tensor & index, const Tensor & value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill_(dim, index, value);
      };
      return wrap(dispatch_index_fill_(self, _r.dimname(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::index_fill_.int_Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)
      auto dispatch_index_fill_ = [](Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill_(dim, index, value);
      };
      return wrap(dispatch_index_fill_(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::index_fill_.Dimname_Scalar(Tensor(a!) self, Dimname dim, Tensor index, Scalar value) -> Tensor(a!)
      auto dispatch_index_fill_ = [](Tensor & self, Dimname dim, const Tensor & index, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill_(dim, index, value);
      };
      return wrap(dispatch_index_fill_(self, _r.dimname(0), _r.tensor(1), _r.scalar(2)));
    }
    case 3: {
      // aten::index_fill_.int_Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
      auto dispatch_index_fill_ = [](Tensor & self, int64_t dim, const Tensor & index, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill_(dim, index, value);
      };
      return wrap(dispatch_index_fill_(self, _r.toInt64(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// index_put
static PyObject * THPVariable_index_put(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "index_put(TensorList? indices, Tensor values, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
  auto dispatch_index_put = [](Tensor & self, TensorList indices, const Tensor & values, bool accumulate) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.index_put(indices, values, accumulate);
  };
  return wrap(dispatch_index_put(self, _r.tensorlist(0), _r.tensor(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// index_put_
static PyObject * THPVariable_index_put_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "index_put_(TensorList? indices, Tensor values, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)
  auto dispatch_index_put_ = [](Tensor & self, TensorList indices, const Tensor & values, bool accumulate) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.index_put_(indices, values, accumulate);
  };
  return wrap(dispatch_index_put_(self, _r.tensorlist(0), _r.tensor(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_select
static PyObject * THPVariable_index_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "index_select(Dimname dim, Tensor index)",
    "index_select(int64_t dim, Tensor index)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::index_select.dimname(Tensor self, Dimname dim, Tensor index) -> Tensor
      auto dispatch_index_select = [](Tensor & self, Dimname dim, const Tensor & index) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_select(dim, index);
      };
      return wrap(dispatch_index_select(self, _r.dimname(0), _r.tensor(1)));
    }
    case 1: {
      // aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
      auto dispatch_index_select = [](Tensor & self, int64_t dim, const Tensor & index) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_select(dim, index);
      };
      return wrap(dispatch_index_select(self, _r.toInt64(0), _r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// indices
static PyObject * THPVariable_indices(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::indices(Tensor(a) self) -> Tensor(a)
  auto dispatch_indices = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.indices();
  };
  return wrap(dispatch_indices(self));
  END_HANDLE_TH_ERRORS
}

// int_repr
static PyObject * THPVariable_int_repr(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::int_repr(Tensor self) -> Tensor
  auto dispatch_int_repr = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.int_repr();
  };
  return wrap(dispatch_int_repr(self));
  END_HANDLE_TH_ERRORS
}

// inverse
static PyObject * THPVariable_inverse(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::inverse(Tensor self) -> Tensor
  auto dispatch_inverse = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.inverse();
  };
  return wrap(dispatch_inverse(self));
  END_HANDLE_TH_ERRORS
}

// irfft
static PyObject * THPVariable_irfft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "irfft(int64_t signal_ndim, bool normalized=False, bool onesided=True, IntArrayRef signal_sizes=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::irfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True, int[] signal_sizes=[]) -> Tensor
  auto dispatch_irfft = [](Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.irfft(signal_ndim, normalized, onesided, signal_sizes);
  };
  return wrap(dispatch_irfft(self, _r.toInt64(0), _r.toBool(1), _r.toBool(2), _r.intlist(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_coalesced
static PyObject * THPVariable_is_coalesced(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::is_coalesced(Tensor self) -> bool
  auto dispatch_is_coalesced = [](Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_coalesced();
  };
  return wrap(dispatch_is_coalesced(self));
  END_HANDLE_TH_ERRORS
}

// is_complex
static PyObject * THPVariable_is_complex(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::is_complex(Tensor self) -> bool
  auto dispatch_is_complex = [](Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_complex();
  };
  return wrap(dispatch_is_complex(self));
  END_HANDLE_TH_ERRORS
}

// is_distributed
static PyObject * THPVariable_is_distributed(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::is_distributed(Tensor self) -> bool
  auto dispatch_is_distributed = [](Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_distributed();
  };
  return wrap(dispatch_is_distributed(self));
  END_HANDLE_TH_ERRORS
}

// is_floating_point
static PyObject * THPVariable_is_floating_point(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::is_floating_point(Tensor self) -> bool
  auto dispatch_is_floating_point = [](Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_floating_point();
  };
  return wrap(dispatch_is_floating_point(self));
  END_HANDLE_TH_ERRORS
}

// is_nonzero
static PyObject * THPVariable_is_nonzero(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::is_nonzero(Tensor self) -> bool
  auto dispatch_is_nonzero = [](Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_nonzero();
  };
  return wrap(dispatch_is_nonzero(self));
  END_HANDLE_TH_ERRORS
}

// is_pinned
static PyObject * THPVariable_is_pinned(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::is_pinned(Tensor self) -> bool
  auto dispatch_is_pinned = [](Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_pinned();
  };
  return wrap(dispatch_is_pinned(self));
  END_HANDLE_TH_ERRORS
}

// is_same_size
static PyObject * THPVariable_is_same_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "is_same_size(Tensor other)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::is_same_size(Tensor self, Tensor other) -> bool
  auto dispatch_is_same_size = [](Tensor & self, const Tensor & other) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_same_size(other);
  };
  return wrap(dispatch_is_same_size(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_set_to
static PyObject * THPVariable_is_set_to(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "is_set_to(Tensor tensor)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::is_set_to(Tensor self, Tensor tensor) -> bool
  auto dispatch_is_set_to = [](Tensor & self, const Tensor & tensor) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_set_to(tensor);
  };
  return wrap(dispatch_is_set_to(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_signed
static PyObject * THPVariable_is_signed(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::is_signed(Tensor self) -> bool
  auto dispatch_is_signed = [](Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_signed();
  };
  return wrap(dispatch_is_signed(self));
  END_HANDLE_TH_ERRORS
}

// isclose
static PyObject * THPVariable_isclose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "isclose(Tensor other, double rtol=1e-05, double atol=1e-08, bool equal_nan=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor
  auto dispatch_isclose = [](Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.isclose(other, rtol, atol, equal_nan);
  };
  return wrap(dispatch_isclose(self, _r.tensor(0), _r.toDouble(1), _r.toDouble(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// kthvalue
static PyObject * THPVariable_kthvalue(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.kthvalue", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "kthvalue(int64_t k, Dimname dim, bool keepdim=False)",
    "kthvalue(int64_t k, int64_t dim=-1, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::kthvalue.dimname(Tensor self, int k, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      auto dispatch_kthvalue = [](Tensor & self, int64_t k, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.kthvalue(k, dim, keepdim);
      };
      return wrap(&NamedTuple, dispatch_kthvalue(self, _r.toInt64(0), _r.dimname(1), _r.toBool(2)));
    }
    case 1: {
      // aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
      auto dispatch_kthvalue = [](Tensor & self, int64_t k, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.kthvalue(k, dim, keepdim);
      };
      return wrap(&NamedTuple, dispatch_kthvalue(self, _r.toInt64(0), _r.toInt64(1), _r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// le
static PyObject * THPVariable_le(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "le(Tensor other)",
    "le(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::le.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch_le = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.le(other);
      };
      return wrap(dispatch_le(self, _r.tensor(0)));
    }
    case 1: {
      // aten::le.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch_le = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.le(other);
      };
      return wrap(dispatch_le(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// le_
static PyObject * THPVariable_le_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "le_(Tensor other)",
    "le_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch_le_ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.le_(other);
      };
      return wrap(dispatch_le_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch_le_ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.le_(other);
      };
      return wrap(dispatch_le_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// lerp
static PyObject * THPVariable_lerp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "lerp(Tensor end, Tensor weight)",
    "lerp(Tensor end, Scalar weight)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor
      auto dispatch_lerp = [](Tensor & self, const Tensor & end, const Tensor & weight) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lerp(end, weight);
      };
      return wrap(dispatch_lerp(self, _r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor
      auto dispatch_lerp = [](Tensor & self, const Tensor & end, Scalar weight) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lerp(end, weight);
      };
      return wrap(dispatch_lerp(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// lerp_
static PyObject * THPVariable_lerp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "lerp_(Tensor end, Tensor weight)",
    "lerp_(Tensor end, Scalar weight)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)
      auto dispatch_lerp_ = [](Tensor & self, const Tensor & end, const Tensor & weight) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lerp_(end, weight);
      };
      return wrap(dispatch_lerp_(self, _r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)
      auto dispatch_lerp_ = [](Tensor & self, const Tensor & end, Scalar weight) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lerp_(end, weight);
      };
      return wrap(dispatch_lerp_(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lgamma
static PyObject * THPVariable_lgamma(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::lgamma(Tensor self) -> Tensor
  auto dispatch_lgamma = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.lgamma();
  };
  return wrap(dispatch_lgamma(self));
  END_HANDLE_TH_ERRORS
}

// lgamma_
static PyObject * THPVariable_lgamma_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::lgamma_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_lgamma_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.lgamma_();
  };
  return wrap(dispatch_lgamma_(self));
  END_HANDLE_TH_ERRORS
}

// log
static PyObject * THPVariable_log(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::log(Tensor self) -> Tensor
  auto dispatch_log = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log();
  };
  return wrap(dispatch_log(self));
  END_HANDLE_TH_ERRORS
}

// log10
static PyObject * THPVariable_log10(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::log10(Tensor self) -> Tensor
  auto dispatch_log10 = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log10();
  };
  return wrap(dispatch_log10(self));
  END_HANDLE_TH_ERRORS
}

// log10_
static PyObject * THPVariable_log10_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::log10_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_log10_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log10_();
  };
  return wrap(dispatch_log10_(self));
  END_HANDLE_TH_ERRORS
}

// log1p
static PyObject * THPVariable_log1p(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::log1p(Tensor self) -> Tensor
  auto dispatch_log1p = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log1p();
  };
  return wrap(dispatch_log1p(self));
  END_HANDLE_TH_ERRORS
}

// log1p_
static PyObject * THPVariable_log1p_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::log1p_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_log1p_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log1p_();
  };
  return wrap(dispatch_log1p_(self));
  END_HANDLE_TH_ERRORS
}

// log2
static PyObject * THPVariable_log2(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::log2(Tensor self) -> Tensor
  auto dispatch_log2 = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log2();
  };
  return wrap(dispatch_log2(self));
  END_HANDLE_TH_ERRORS
}

// log2_
static PyObject * THPVariable_log2_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::log2_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_log2_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log2_();
  };
  return wrap(dispatch_log2_(self));
  END_HANDLE_TH_ERRORS
}

// log_
static PyObject * THPVariable_log_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::log_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_log_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log_();
  };
  return wrap(dispatch_log_(self));
  END_HANDLE_TH_ERRORS
}

// log_normal_
static PyObject * THPVariable_log_normal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "log_normal_(double mean=1, double std=2, *, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)
  auto dispatch_log_normal_ = [](Tensor & self, double mean, double std, Generator * generator) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log_normal_(mean, std, generator);
  };
  return wrap(dispatch_log_normal_(self, _r.toDouble(0), _r.toDouble(1), _r.generator(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// log_softmax
static PyObject * THPVariable_log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "log_softmax(Dimname dim, *, ScalarType? dtype=None)",
    "log_softmax(int64_t dim, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_log_softmax = [](Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.log_softmax(dim, dtype);
      };
      return wrap(dispatch_log_softmax(self, _r.dimname(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      // aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
      auto dispatch_log_softmax = [](Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.log_softmax(dim, dtype);
      };
      return wrap(dispatch_log_softmax(self, _r.toInt64(0), _r.scalartypeOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logdet
static PyObject * THPVariable_logdet(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::logdet(Tensor self) -> Tensor
  auto dispatch_logdet = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logdet();
  };
  return wrap(dispatch_logdet(self));
  END_HANDLE_TH_ERRORS
}

// logical_and
static PyObject * THPVariable_logical_and(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "logical_and(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::logical_and(Tensor self, Tensor other) -> Tensor
  auto dispatch_logical_and = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_and(other);
  };
  return wrap(dispatch_logical_and(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_and_
static PyObject * THPVariable_logical_and_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "logical_and_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::logical_and_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  auto dispatch_logical_and_ = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_and_(other);
  };
  return wrap(dispatch_logical_and_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_not
static PyObject * THPVariable_logical_not(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::logical_not(Tensor self) -> Tensor
  auto dispatch_logical_not = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_not();
  };
  return wrap(dispatch_logical_not(self));
  END_HANDLE_TH_ERRORS
}

// logical_not_
static PyObject * THPVariable_logical_not_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::logical_not_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_logical_not_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_not_();
  };
  return wrap(dispatch_logical_not_(self));
  END_HANDLE_TH_ERRORS
}

// logical_or
static PyObject * THPVariable_logical_or(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "logical_or(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::logical_or(Tensor self, Tensor other) -> Tensor
  auto dispatch_logical_or = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_or(other);
  };
  return wrap(dispatch_logical_or(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_or_
static PyObject * THPVariable_logical_or_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "logical_or_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::logical_or_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  auto dispatch_logical_or_ = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_or_(other);
  };
  return wrap(dispatch_logical_or_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_xor
static PyObject * THPVariable_logical_xor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "logical_xor(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::logical_xor(Tensor self, Tensor other) -> Tensor
  auto dispatch_logical_xor = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_xor(other);
  };
  return wrap(dispatch_logical_xor(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_xor_
static PyObject * THPVariable_logical_xor_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "logical_xor_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::logical_xor_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  auto dispatch_logical_xor_ = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_xor_(other);
  };
  return wrap(dispatch_logical_xor_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// logsumexp
static PyObject * THPVariable_logsumexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "logsumexp(DimnameList[1] dim, bool keepdim=False)",
    "logsumexp(IntArrayRef[1] dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::logsumexp.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor
      auto dispatch_logsumexp = [](Tensor & self, DimnameList dim, bool keepdim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.logsumexp(dim, keepdim);
      };
      return wrap(dispatch_logsumexp(self, _r.dimnamelist(0), _r.toBool(1)));
    }
    case 1: {
      // aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
      auto dispatch_logsumexp = [](Tensor & self, IntArrayRef dim, bool keepdim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.logsumexp(dim, keepdim);
      };
      return wrap(dispatch_logsumexp(self, _r.intlist(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lstsq
static PyObject * THPVariable_lstsq(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"solution", ""}, {"QR", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.lstsq", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "lstsq(Tensor A)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::lstsq(Tensor self, Tensor A) -> (Tensor solution, Tensor QR)
  auto dispatch_lstsq = [](Tensor & self, const Tensor & A) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.lstsq(A);
  };
  return wrap(&NamedTuple, dispatch_lstsq(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// lt
static PyObject * THPVariable_lt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "lt(Tensor other)",
    "lt(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::lt.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch_lt = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lt(other);
      };
      return wrap(dispatch_lt(self, _r.tensor(0)));
    }
    case 1: {
      // aten::lt.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch_lt = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lt(other);
      };
      return wrap(dispatch_lt(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// lt_
static PyObject * THPVariable_lt_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "lt_(Tensor other)",
    "lt_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch_lt_ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lt_(other);
      };
      return wrap(dispatch_lt_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch_lt_ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lt_(other);
      };
      return wrap(dispatch_lt_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lu_solve
static PyObject * THPVariable_lu_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "lu_solve(Tensor LU_data, Tensor LU_pivots)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor
  auto dispatch_lu_solve = [](Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.lu_solve(LU_data, LU_pivots);
  };
  return wrap(dispatch_lu_solve(self, _r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// masked_fill
static PyObject * THPVariable_masked_fill(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "masked_fill(Tensor mask, Tensor value)",
    "masked_fill(Tensor mask, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
      auto dispatch_masked_fill = [](Tensor & self, const Tensor & mask, const Tensor & value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.masked_fill(mask, value);
      };
      return wrap(dispatch_masked_fill(self, _r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
      auto dispatch_masked_fill = [](Tensor & self, const Tensor & mask, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.masked_fill(mask, value);
      };
      return wrap(dispatch_masked_fill(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// masked_fill_
static PyObject * THPVariable_masked_fill_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "masked_fill_(Tensor mask, Tensor value)",
    "masked_fill_(Tensor mask, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)
      auto dispatch_masked_fill_ = [](Tensor & self, const Tensor & mask, const Tensor & value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.masked_fill_(mask, value);
      };
      return wrap(dispatch_masked_fill_(self, _r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)
      auto dispatch_masked_fill_ = [](Tensor & self, const Tensor & mask, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.masked_fill_(mask, value);
      };
      return wrap(dispatch_masked_fill_(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// masked_scatter
static PyObject * THPVariable_masked_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "masked_scatter(Tensor mask, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
  auto dispatch_masked_scatter = [](Tensor & self, const Tensor & mask, const Tensor & source) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.masked_scatter(mask, source);
  };
  return wrap(dispatch_masked_scatter(self, _r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// masked_scatter_
static PyObject * THPVariable_masked_scatter_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "masked_scatter_(Tensor mask, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)
  auto dispatch_masked_scatter_ = [](Tensor & self, const Tensor & mask, const Tensor & source) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.masked_scatter_(mask, source);
  };
  return wrap(dispatch_masked_scatter_(self, _r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// masked_select
static PyObject * THPVariable_masked_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "masked_select(Tensor mask)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::masked_select(Tensor self, Tensor mask) -> Tensor
  auto dispatch_masked_select = [](Tensor & self, const Tensor & mask) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.masked_select(mask);
  };
  return wrap(dispatch_masked_select(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// matmul
static PyObject * THPVariable_matmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "matmul(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::matmul(Tensor self, Tensor other) -> Tensor
  auto dispatch_matmul = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.matmul(other);
  };
  return wrap(dispatch_matmul(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// matrix_power
static PyObject * THPVariable_matrix_power(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "matrix_power(int64_t n)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::matrix_power(Tensor self, int n) -> Tensor
  auto dispatch_matrix_power = [](Tensor & self, int64_t n) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.matrix_power(n);
  };
  return wrap(dispatch_matrix_power(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// max
static PyObject * THPVariable_max(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.max", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "max()",
    "max(Dimname dim, bool keepdim=False)",
    "max(Tensor other)",
    "max(int64_t dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::max(Tensor self) -> Tensor
      auto dispatch_max = [](Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.max();
      };
      return wrap(dispatch_max(self));
    }
    case 1: {
      // aten::max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      auto dispatch_max = [](Tensor & self, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.max(dim, keepdim);
      };
      return wrap(&NamedTuple, dispatch_max(self, _r.dimname(0), _r.toBool(1)));
    }
    case 2: {
      // aten::max.other(Tensor self, Tensor other) -> Tensor
      auto dispatch_max = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.max(other);
      };
      return wrap(dispatch_max(self, _r.tensor(0)));
    }
    case 3: {
      // aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      auto dispatch_max = [](Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.max(dim, keepdim);
      };
      return wrap(&NamedTuple, dispatch_max(self, _r.toInt64(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// mean
static PyObject * THPVariable_mean(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "mean(*, ScalarType? dtype=None)",
    "mean(DimnameList[1] dim, bool keepdim=False, *, ScalarType? dtype=None)",
    "mean(IntArrayRef[1] dim, bool keepdim=False, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_mean = [](Tensor & self, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.mean(dtype);
      };
      return wrap(dispatch_mean(self, _r.scalartypeOptional(0)));
    }
    case 1: {
      // aten::mean.names_dim(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_mean = [](Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.mean(dim, keepdim, dtype);
      };
      return wrap(dispatch_mean(self, _r.dimnamelist(0), _r.toBool(1), _r.scalartypeOptional(2)));
    }
    case 2: {
      // aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_mean = [](Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.mean(dim, keepdim, dtype);
      };
      return wrap(dispatch_mean(self, _r.intlist(0), _r.toBool(1), _r.scalartypeOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// median
static PyObject * THPVariable_median(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.median", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "median()",
    "median(Dimname dim, bool keepdim=False)",
    "median(int64_t dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::median(Tensor self) -> Tensor
      auto dispatch_median = [](Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.median();
      };
      return wrap(dispatch_median(self));
    }
    case 1: {
      // aten::median.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      auto dispatch_median = [](Tensor & self, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.median(dim, keepdim);
      };
      return wrap(&NamedTuple, dispatch_median(self, _r.dimname(0), _r.toBool(1)));
    }
    case 2: {
      // aten::median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      auto dispatch_median = [](Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.median(dim, keepdim);
      };
      return wrap(&NamedTuple, dispatch_median(self, _r.toInt64(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// min
static PyObject * THPVariable_min(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.min", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "min()",
    "min(Dimname dim, bool keepdim=False)",
    "min(Tensor other)",
    "min(int64_t dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::min(Tensor self) -> Tensor
      auto dispatch_min = [](Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.min();
      };
      return wrap(dispatch_min(self));
    }
    case 1: {
      // aten::min.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      auto dispatch_min = [](Tensor & self, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.min(dim, keepdim);
      };
      return wrap(&NamedTuple, dispatch_min(self, _r.dimname(0), _r.toBool(1)));
    }
    case 2: {
      // aten::min.other(Tensor self, Tensor other) -> Tensor
      auto dispatch_min = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.min(other);
      };
      return wrap(dispatch_min(self, _r.tensor(0)));
    }
    case 3: {
      // aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      auto dispatch_min = [](Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.min(dim, keepdim);
      };
      return wrap(&NamedTuple, dispatch_min(self, _r.toInt64(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mm
static PyObject * THPVariable_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "mm(Tensor mat2)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::mm(Tensor self, Tensor mat2) -> Tensor
  auto dispatch_mm = [](Tensor & self, const Tensor & mat2) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.mm(mat2);
  };
  return wrap(dispatch_mm(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// mode
static PyObject * THPVariable_mode(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.mode", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "mode(Dimname dim, bool keepdim=False)",
    "mode(int64_t dim=-1, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::mode.dimname(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      auto dispatch_mode = [](Tensor & self, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.mode(dim, keepdim);
      };
      return wrap(&NamedTuple, dispatch_mode(self, _r.dimname(0), _r.toBool(1)));
    }
    case 1: {
      // aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
      auto dispatch_mode = [](Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.mode(dim, keepdim);
      };
      return wrap(&NamedTuple, dispatch_mode(self, _r.toInt64(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mul
static PyObject * THPVariable_mul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "mul(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
  auto dispatch_mul = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.mul(other);
  };
  return wrap(dispatch_mul(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mul_
static PyObject * THPVariable_mul_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "mul_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
  auto dispatch_mul_ = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.mul_(other);
  };
  return wrap(dispatch_mul_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// multinomial
static PyObject * THPVariable_multinomial(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "multinomial(int64_t num_samples, bool replacement=False, *, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor
  auto dispatch_multinomial = [](Tensor & self, int64_t num_samples, bool replacement, Generator * generator) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.multinomial(num_samples, replacement, generator);
  };
  return wrap(dispatch_multinomial(self, _r.toInt64(0), _r.toBool(1), _r.generator(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mv
static PyObject * THPVariable_mv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "mv(Tensor vec)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::mv(Tensor self, Tensor vec) -> Tensor
  auto dispatch_mv = [](Tensor & self, const Tensor & vec) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.mv(vec);
  };
  return wrap(dispatch_mv(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mvlgamma
static PyObject * THPVariable_mvlgamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "mvlgamma(int64_t p)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::mvlgamma(Tensor self, int p) -> Tensor
  auto dispatch_mvlgamma = [](Tensor & self, int64_t p) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.mvlgamma(p);
  };
  return wrap(dispatch_mvlgamma(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mvlgamma_
static PyObject * THPVariable_mvlgamma_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "mvlgamma_(int64_t p)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)
  auto dispatch_mvlgamma_ = [](Tensor & self, int64_t p) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.mvlgamma_(p);
  };
  return wrap(dispatch_mvlgamma_(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// narrow
static PyObject * THPVariable_narrow(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "narrow(int64_t dim, int64_t start, int64_t length)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)
  auto dispatch_narrow = [](Tensor & self, int64_t dim, int64_t start, int64_t length) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.narrow(dim, start, length);
  };
  return wrap(dispatch_narrow(self, _r.toInt64(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// narrow_copy
static PyObject * THPVariable_narrow_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "narrow_copy(int64_t dim, int64_t start, int64_t length)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::narrow_copy(Tensor self, int dim, int start, int length) -> Tensor
  auto dispatch_narrow_copy = [](Tensor & self, int64_t dim, int64_t start, int64_t length) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.narrow_copy(dim, start, length);
  };
  return wrap(dispatch_narrow_copy(self, _r.toInt64(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ne
static PyObject * THPVariable_ne(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "ne(Tensor other)",
    "ne(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::ne.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch_ne = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ne(other);
      };
      return wrap(dispatch_ne(self, _r.tensor(0)));
    }
    case 1: {
      // aten::ne.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch_ne = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ne(other);
      };
      return wrap(dispatch_ne(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ne_
static PyObject * THPVariable_ne_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "ne_(Tensor other)",
    "ne_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch_ne_ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ne_(other);
      };
      return wrap(dispatch_ne_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch_ne_ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ne_(other);
      };
      return wrap(dispatch_ne_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// neg
static PyObject * THPVariable_neg(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::neg(Tensor self) -> Tensor
  auto dispatch_neg = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.neg();
  };
  return wrap(dispatch_neg(self));
  END_HANDLE_TH_ERRORS
}

// neg_
static PyObject * THPVariable_neg_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::neg_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_neg_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.neg_();
  };
  return wrap(dispatch_neg_(self));
  END_HANDLE_TH_ERRORS
}

// new_empty
static PyObject * THPVariable_new_empty(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "new_empty(IntArrayRef size, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::new_empty(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeWithDefault(1, self.scalar_type()))
      .device(_r.deviceWithDefault(3, self.device()))
      .layout(_r.layoutWithDefault(2, *torch::getLayout(self.options().backend())).layout)
      .requires_grad(_r.toBool(5))
      .pinned_memory(_r.toBool(4));
  auto dtype = _r.scalartypeWithDefault(1, self.scalar_type());
  auto layout = _r.layoutWithDefault(2, *torch::getLayout(self.options().backend())).layout;
  auto device = _r.deviceWithDefault(3, self.device());
  auto pin_memory = _r.toBool(4);
  auto requires_grad = _r.toBool(5);
  torch::utils::maybe_initialize_cuda(options);
  auto dispatch_new_empty = [](Tensor & self, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._new_empty(size, dtype, layout, device, pin_memory);
  };
  return wrap(dispatch_new_empty(self, _r.intlist(0), dtype, layout, device, pin_memory));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// new_full
static PyObject * THPVariable_new_full(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "new_full(IntArrayRef size, Scalar fill_value, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::new_full(Tensor self, int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeWithDefault(2, self.scalar_type()))
      .device(_r.deviceWithDefault(4, self.device()))
      .layout(_r.layoutWithDefault(3, *torch::getLayout(self.options().backend())).layout)
      .requires_grad(_r.toBool(6))
      .pinned_memory(_r.toBool(5));
  auto dtype = _r.scalartypeWithDefault(2, self.scalar_type());
  auto layout = _r.layoutWithDefault(3, *torch::getLayout(self.options().backend())).layout;
  auto device = _r.deviceWithDefault(4, self.device());
  auto pin_memory = _r.toBool(5);
  auto requires_grad = _r.toBool(6);
  torch::utils::maybe_initialize_cuda(options);
  auto dispatch_new_full = [](Tensor & self, IntArrayRef size, Scalar fill_value, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._new_full(size, fill_value, dtype, layout, device, pin_memory);
  };
  return wrap(dispatch_new_full(self, _r.intlist(0), _r.scalar(1), dtype, layout, device, pin_memory));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// new_zeros
static PyObject * THPVariable_new_zeros(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "new_zeros(IntArrayRef size, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::new_zeros(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeWithDefault(1, self.scalar_type()))
      .device(_r.deviceWithDefault(3, self.device()))
      .layout(_r.layoutWithDefault(2, *torch::getLayout(self.options().backend())).layout)
      .requires_grad(_r.toBool(5))
      .pinned_memory(_r.toBool(4));
  auto dtype = _r.scalartypeWithDefault(1, self.scalar_type());
  auto layout = _r.layoutWithDefault(2, *torch::getLayout(self.options().backend())).layout;
  auto device = _r.deviceWithDefault(3, self.device());
  auto pin_memory = _r.toBool(4);
  auto requires_grad = _r.toBool(5);
  torch::utils::maybe_initialize_cuda(options);
  auto dispatch_new_zeros = [](Tensor & self, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._new_zeros(size, dtype, layout, device, pin_memory);
  };
  return wrap(dispatch_new_zeros(self, _r.intlist(0), dtype, layout, device, pin_memory));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// norm
static PyObject * THPVariable_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "norm(Scalar p=2)",
    "norm(Scalar? p, *, ScalarType dtype)",
    "norm(Scalar? p, DimnameList[1] dim, bool keepdim, *, ScalarType dtype)",
    "norm(Scalar? p, DimnameList[1] dim, bool keepdim=False)",
    "norm(Scalar? p, IntArrayRef[1] dim, bool keepdim, *, ScalarType dtype)",
    "norm(Scalar? p, IntArrayRef[1] dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor
      auto dispatch_norm = [](Tensor & self, Scalar p) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p);
      };
      return wrap(dispatch_norm(self, _r.scalar(0)));
    }
    case 1: {
      // aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor
      auto dispatch_norm = [](Tensor & self, c10::optional<Scalar> p, ScalarType dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p, dtype);
      };
      return wrap(dispatch_norm(self, _r.scalarOptional(0), _r.scalartype(1)));
    }
    case 2: {
      // aten::norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
      auto dispatch_norm = [](Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim, ScalarType dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p, dim, keepdim, dtype);
      };
      return wrap(dispatch_norm(self, _r.scalarOptional(0), _r.dimnamelist(1), _r.toBool(2), _r.scalartype(3)));
    }
    case 3: {
      // aten::norm.names_ScalarOpt_dim(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False) -> Tensor
      auto dispatch_norm = [](Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p, dim, keepdim);
      };
      return wrap(dispatch_norm(self, _r.scalarOptional(0), _r.dimnamelist(1), _r.toBool(2)));
    }
    case 4: {
      // aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
      auto dispatch_norm = [](Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p, dim, keepdim, dtype);
      };
      return wrap(dispatch_norm(self, _r.scalarOptional(0), _r.intlist(1), _r.toBool(2), _r.scalartype(3)));
    }
    case 5: {
      // aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor
      auto dispatch_norm = [](Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p, dim, keepdim);
      };
      return wrap(dispatch_norm(self, _r.scalarOptional(0), _r.intlist(1), _r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// normal_
static PyObject * THPVariable_normal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "normal_(double mean=0, double std=1, *, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)
  auto dispatch_normal_ = [](Tensor & self, double mean, double std, Generator * generator) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.normal_(mean, std, generator);
  };
  return wrap(dispatch_normal_(self, _r.toDouble(0), _r.toDouble(1), _r.generator(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// orgqr
static PyObject * THPVariable_orgqr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "orgqr(Tensor input2)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::orgqr(Tensor self, Tensor input2) -> Tensor
  auto dispatch_orgqr = [](Tensor & self, const Tensor & input2) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.orgqr(input2);
  };
  return wrap(dispatch_orgqr(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ormqr
static PyObject * THPVariable_ormqr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "ormqr(Tensor input2, Tensor input3, bool left=True, bool transpose=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor
  auto dispatch_ormqr = [](Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ormqr(input2, input3, left, transpose);
  };
  return wrap(dispatch_ormqr(self, _r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// permute
static PyObject * THPVariable_permute(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "permute(IntArrayRef dims)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
  auto dispatch_permute = [](Tensor & self, IntArrayRef dims) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.permute(dims);
  };
  return wrap(dispatch_permute(self, _r.intlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// pin_memory
static PyObject * THPVariable_pin_memory(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::pin_memory(Tensor self) -> Tensor
  auto dispatch_pin_memory = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.pin_memory();
  };
  return wrap(dispatch_pin_memory(self));
  END_HANDLE_TH_ERRORS
}

// pinverse
static PyObject * THPVariable_pinverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "pinverse(double rcond=1e-15)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::pinverse(Tensor self, float rcond=1e-15) -> Tensor
  auto dispatch_pinverse = [](Tensor & self, double rcond) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.pinverse(rcond);
  };
  return wrap(dispatch_pinverse(self, _r.toDouble(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// polygamma
static PyObject * THPVariable_polygamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "polygamma(int64_t n)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::polygamma(int n, Tensor self) -> Tensor
  auto dispatch_polygamma = [](int64_t n, Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.polygamma(n);
  };
  return wrap(dispatch_polygamma(_r.toInt64(0), self));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// polygamma_
static PyObject * THPVariable_polygamma_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "polygamma_(int64_t n)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::polygamma_(Tensor(a!) self, int n) -> Tensor(a!)
  auto dispatch_polygamma_ = [](Tensor & self, int64_t n) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.polygamma_(n);
  };
  return wrap(dispatch_polygamma_(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// pow
static PyObject * THPVariable_pow(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "pow(Tensor exponent)",
    "pow(Scalar exponent)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
      auto dispatch_pow = [](Tensor & self, const Tensor & exponent) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.pow(exponent);
      };
      return wrap(dispatch_pow(self, _r.tensor(0)));
    }
    case 1: {
      // aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
      auto dispatch_pow = [](Tensor & self, Scalar exponent) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.pow(exponent);
      };
      return wrap(dispatch_pow(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// pow_
static PyObject * THPVariable_pow_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "pow_(Tensor exponent)",
    "pow_(Scalar exponent)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)
      auto dispatch_pow_ = [](Tensor & self, const Tensor & exponent) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.pow_(exponent);
      };
      return wrap(dispatch_pow_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)
      auto dispatch_pow_ = [](Tensor & self, Scalar exponent) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.pow_(exponent);
      };
      return wrap(dispatch_pow_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// prelu
static PyObject * THPVariable_prelu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "prelu(Tensor weight)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::prelu(Tensor self, Tensor weight) -> Tensor
  auto dispatch_prelu = [](Tensor & self, const Tensor & weight) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.prelu(weight);
  };
  return wrap(dispatch_prelu(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// prod
static PyObject * THPVariable_prod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "prod(*, ScalarType? dtype=None)",
    "prod(Dimname dim, bool keepdim=False, *, ScalarType? dtype=None)",
    "prod(int64_t dim, bool keepdim=False, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_prod = [](Tensor & self, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.prod(dtype);
      };
      return wrap(dispatch_prod(self, _r.scalartypeOptional(0)));
    }
    case 1: {
      // aten::prod.dim_Dimname(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_prod = [](Tensor & self, Dimname dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.prod(dim, keepdim, dtype);
      };
      return wrap(dispatch_prod(self, _r.dimname(0), _r.toBool(1), _r.scalartypeOptional(2)));
    }
    case 2: {
      // aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_prod = [](Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.prod(dim, keepdim, dtype);
      };
      return wrap(dispatch_prod(self, _r.toInt64(0), _r.toBool(1), _r.scalartypeOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// put_
static PyObject * THPVariable_put_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "put_(Tensor index, Tensor source, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)
  auto dispatch_put_ = [](Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.put_(index, source, accumulate);
  };
  return wrap(dispatch_put_(self, _r.tensor(0), _r.tensor(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// q_per_channel_axis
static PyObject * THPVariable_q_per_channel_axis(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::q_per_channel_axis(Tensor self) -> int
  auto dispatch_q_per_channel_axis = [](Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self.q_per_channel_axis();
  };
  return wrap(dispatch_q_per_channel_axis(self));
  END_HANDLE_TH_ERRORS
}

// q_per_channel_scales
static PyObject * THPVariable_q_per_channel_scales(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::q_per_channel_scales(Tensor self) -> Tensor
  auto dispatch_q_per_channel_scales = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.q_per_channel_scales();
  };
  return wrap(dispatch_q_per_channel_scales(self));
  END_HANDLE_TH_ERRORS
}

// q_per_channel_zero_points
static PyObject * THPVariable_q_per_channel_zero_points(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::q_per_channel_zero_points(Tensor self) -> Tensor
  auto dispatch_q_per_channel_zero_points = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.q_per_channel_zero_points();
  };
  return wrap(dispatch_q_per_channel_zero_points(self));
  END_HANDLE_TH_ERRORS
}

// q_scale
static PyObject * THPVariable_q_scale(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::q_scale(Tensor self) -> float
  auto dispatch_q_scale = [](Tensor & self) -> double {
    pybind11::gil_scoped_release no_gil;
    return self.q_scale();
  };
  return wrap(dispatch_q_scale(self));
  END_HANDLE_TH_ERRORS
}

// q_zero_point
static PyObject * THPVariable_q_zero_point(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::q_zero_point(Tensor self) -> int
  auto dispatch_q_zero_point = [](Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self.q_zero_point();
  };
  return wrap(dispatch_q_zero_point(self));
  END_HANDLE_TH_ERRORS
}

// qr
static PyObject * THPVariable_qr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"Q", ""}, {"R", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.qr", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "qr(bool some=True)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)
  auto dispatch_qr = [](Tensor & self, bool some) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.qr(some);
  };
  return wrap(&NamedTuple, dispatch_qr(self, _r.toBool(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// qscheme
static PyObject * THPVariable_qscheme(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::qscheme(Tensor self) -> QScheme
  auto dispatch_qscheme = [](Tensor & self) -> QScheme {
    pybind11::gil_scoped_release no_gil;
    return self.qscheme();
  };
  return wrap(dispatch_qscheme(self));
  END_HANDLE_TH_ERRORS
}

\
// random_
static PyObject * THPVariable_random_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "random_(*, Generator generator=None)",
    "random_(int64_t from, int64_t to, *, Generator generator=None)",
    "random_(int64_t to, *, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)
      auto dispatch_random_ = [](Tensor & self, Generator * generator) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.random_(generator);
      };
      return wrap(dispatch_random_(self, _r.generator(0)));
    }
    case 1: {
      // aten::random_.from(Tensor(a!) self, int from, int to, *, Generator? generator=None) -> Tensor(a!)
      auto dispatch_random_ = [](Tensor & self, int64_t from, int64_t to, Generator * generator) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.random_(from, to, generator);
      };
      return wrap(dispatch_random_(self, _r.toInt64(0), _r.toInt64(1), _r.generator(2)));
    }
    case 2: {
      // aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)
      auto dispatch_random_ = [](Tensor & self, int64_t to, Generator * generator) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.random_(to, generator);
      };
      return wrap(dispatch_random_(self, _r.toInt64(0), _r.generator(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// real
static PyObject * THPVariable_real(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::real(Tensor self) -> Tensor
  auto dispatch_real = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.real();
  };
  return wrap(dispatch_real(self));
  END_HANDLE_TH_ERRORS
}

// reciprocal
static PyObject * THPVariable_reciprocal(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::reciprocal(Tensor self) -> Tensor
  auto dispatch_reciprocal = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.reciprocal();
  };
  return wrap(dispatch_reciprocal(self));
  END_HANDLE_TH_ERRORS
}

// reciprocal_
static PyObject * THPVariable_reciprocal_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_reciprocal_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.reciprocal_();
  };
  return wrap(dispatch_reciprocal_(self));
  END_HANDLE_TH_ERRORS
}

// refine_names
static PyObject * THPVariable_refine_names(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "refine_names(DimnameList names)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::refine_names(Tensor(a) self, Dimname[] names) -> Tensor(a)
  auto dispatch_refine_names = [](Tensor & self, DimnameList names) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.refine_names(names);
  };
  return wrap(dispatch_refine_names(self, _r.dimnamelist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// relu
static PyObject * THPVariable_relu(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::relu(Tensor self) -> Tensor
  auto dispatch_relu = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.relu();
  };
  return wrap(dispatch_relu(self));
  END_HANDLE_TH_ERRORS
}

// relu_
static PyObject * THPVariable_relu_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::relu_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_relu_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.relu_();
  };
  return wrap(dispatch_relu_(self));
  END_HANDLE_TH_ERRORS
}

\
// remainder
static PyObject * THPVariable_remainder(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "remainder(Tensor other)",
    "remainder(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch_remainder = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.remainder(other);
      };
      return wrap(dispatch_remainder(self, _r.tensor(0)));
    }
    case 1: {
      // aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch_remainder = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.remainder(other);
      };
      return wrap(dispatch_remainder(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// remainder_
static PyObject * THPVariable_remainder_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "remainder_(Tensor other)",
    "remainder_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      auto dispatch_remainder_ = [](Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.remainder_(other);
      };
      return wrap(dispatch_remainder_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      auto dispatch_remainder_ = [](Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.remainder_(other);
      };
      return wrap(dispatch_remainder_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rename
static PyObject * THPVariable_rename(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "rename(DimnameList? names)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::rename(Tensor(a) self, Dimname[]? names) -> Tensor(a)
  auto __names = _r.toDimnameListOptional(0);
  c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
  auto dispatch_rename = [](Tensor & self, c10::optional<DimnameList> names) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rename(names);
  };
  return wrap(dispatch_rename(self, names));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rename_
static PyObject * THPVariable_rename_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "rename_(DimnameList? names)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::rename_(Tensor(a!) self, Dimname[]? names) -> Tensor(a!)
  auto __names = _r.toDimnameListOptional(0);
  c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
  auto dispatch_rename_ = [](Tensor & self, c10::optional<DimnameList> names) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rename_(names);
  };
  return wrap(dispatch_rename_(self, names));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// renorm
static PyObject * THPVariable_renorm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "renorm(Scalar p, int64_t dim, Scalar maxnorm)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor
  auto dispatch_renorm = [](Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.renorm(p, dim, maxnorm);
  };
  return wrap(dispatch_renorm(self, _r.scalar(0), _r.toInt64(1), _r.scalar(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// renorm_
static PyObject * THPVariable_renorm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "renorm_(Scalar p, int64_t dim, Scalar maxnorm)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)
  auto dispatch_renorm_ = [](Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.renorm_(p, dim, maxnorm);
  };
  return wrap(dispatch_renorm_(self, _r.scalar(0), _r.toInt64(1), _r.scalar(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// repeat
static PyObject * THPVariable_repeat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "repeat(IntArrayRef repeats)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::repeat(Tensor self, int[] repeats) -> Tensor
  auto dispatch_repeat = [](Tensor & self, IntArrayRef repeats) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.repeat(repeats);
  };
  return wrap(dispatch_repeat(self, _r.intlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// repeat_interleave
static PyObject * THPVariable_repeat_interleave(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "repeat_interleave(Tensor repeats, int64_t? dim=None)",
    "repeat_interleave(int64_t repeats, int64_t? dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None) -> Tensor
      auto dispatch_repeat_interleave = [](Tensor & self, const Tensor & repeats, c10::optional<int64_t> dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.repeat_interleave(repeats, dim);
      };
      return wrap(dispatch_repeat_interleave(self, _r.tensor(0), _r.toInt64Optional(1)));
    }
    case 1: {
      // aten::repeat_interleave.self_int(Tensor self, int repeats, int? dim=None) -> Tensor
      auto dispatch_repeat_interleave = [](Tensor & self, int64_t repeats, c10::optional<int64_t> dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.repeat_interleave(repeats, dim);
      };
      return wrap(dispatch_repeat_interleave(self, _r.toInt64(0), _r.toInt64Optional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reshape
static PyObject * THPVariable_reshape(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "reshape(IntArrayRef shape)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::reshape(Tensor self, int[] shape) -> Tensor
  auto dispatch_reshape = [](Tensor & self, IntArrayRef shape) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.reshape(shape);
  };
  return wrap(dispatch_reshape(self, _r.intlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reshape_as
static PyObject * THPVariable_reshape_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "reshape_as(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::reshape_as(Tensor self, Tensor other) -> Tensor
  auto dispatch_reshape_as = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.reshape_as(other);
  };
  return wrap(dispatch_reshape_as(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// resize_
static PyObject * THPVariable_resize_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "resize_(IntArrayRef size, *, MemoryFormat? memory_format=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::resize_(Tensor(a!) self, int[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)
  auto dispatch_resize_ = [](Tensor & self, IntArrayRef size, c10::optional<MemoryFormat> memory_format) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.resize_(size, memory_format);
  };
  return wrap(dispatch_resize_(self, _r.intlist(0), _r.memoryformatOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// resize_as_
static PyObject * THPVariable_resize_as_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "resize_as_(Tensor the_template, *, MemoryFormat? memory_format=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)
  auto dispatch_resize_as_ = [](Tensor & self, const Tensor & the_template, c10::optional<MemoryFormat> memory_format) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.resize_as_(the_template, memory_format);
  };
  return wrap(dispatch_resize_as_(self, _r.tensor(0), _r.memoryformatOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rfft
static PyObject * THPVariable_rfft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "rfft(int64_t signal_ndim, bool normalized=False, bool onesided=True)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::rfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True) -> Tensor
  auto dispatch_rfft = [](Tensor & self, int64_t signal_ndim, bool normalized, bool onesided) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rfft(signal_ndim, normalized, onesided);
  };
  return wrap(dispatch_rfft(self, _r.toInt64(0), _r.toBool(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// roll
static PyObject * THPVariable_roll(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "roll(IntArrayRef[1] shifts, IntArrayRef[1] dims=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor
  auto dispatch_roll = [](Tensor & self, IntArrayRef shifts, IntArrayRef dims) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.roll(shifts, dims);
  };
  return wrap(dispatch_roll(self, _r.intlist(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rot90
static PyObject * THPVariable_rot90(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "rot90(int64_t k=1, IntArrayRef dims={0,1})",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor
  auto dispatch_rot90 = [](Tensor & self, int64_t k, IntArrayRef dims) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rot90(k, dims);
  };
  return wrap(dispatch_rot90(self, _r.toInt64(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// round
static PyObject * THPVariable_round(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::round(Tensor self) -> Tensor
  auto dispatch_round = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.round();
  };
  return wrap(dispatch_round(self));
  END_HANDLE_TH_ERRORS
}

// round_
static PyObject * THPVariable_round_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::round_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_round_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.round_();
  };
  return wrap(dispatch_round_(self));
  END_HANDLE_TH_ERRORS
}

// rsqrt
static PyObject * THPVariable_rsqrt(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::rsqrt(Tensor self) -> Tensor
  auto dispatch_rsqrt = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rsqrt();
  };
  return wrap(dispatch_rsqrt(self));
  END_HANDLE_TH_ERRORS
}

// rsqrt_
static PyObject * THPVariable_rsqrt_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_rsqrt_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rsqrt_();
  };
  return wrap(dispatch_rsqrt_(self));
  END_HANDLE_TH_ERRORS
}

\
// scatter
static PyObject * THPVariable_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "scatter(Dimname dim, Tensor index, Tensor src)",
    "scatter(int64_t dim, Tensor index, Tensor src)",
    "scatter(Dimname dim, Tensor index, Scalar value)",
    "scatter(int64_t dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::scatter.dimname_src(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
      auto dispatch_scatter = [](Tensor & self, Dimname dim, const Tensor & index, const Tensor & src) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, src);
      };
      return wrap(dispatch_scatter(self, _r.dimname(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
      auto dispatch_scatter = [](Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, src);
      };
      return wrap(dispatch_scatter(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::scatter.dimname_value(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
      auto dispatch_scatter = [](Tensor & self, Dimname dim, const Tensor & index, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, value);
      };
      return wrap(dispatch_scatter(self, _r.dimname(0), _r.tensor(1), _r.scalar(2)));
    }
    case 3: {
      // aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
      auto dispatch_scatter = [](Tensor & self, int64_t dim, const Tensor & index, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, value);
      };
      return wrap(dispatch_scatter(self, _r.toInt64(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// scatter_
static PyObject * THPVariable_scatter_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "scatter_(int64_t dim, Tensor index, Tensor src)",
    "scatter_(int64_t dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
      auto dispatch_scatter_ = [](Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter_(dim, index, src);
      };
      return wrap(dispatch_scatter_(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
      auto dispatch_scatter_ = [](Tensor & self, int64_t dim, const Tensor & index, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter_(dim, index, value);
      };
      return wrap(dispatch_scatter_(self, _r.toInt64(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// scatter_add
static PyObject * THPVariable_scatter_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "scatter_add(Dimname dim, Tensor index, Tensor src)",
    "scatter_add(int64_t dim, Tensor index, Tensor src)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::scatter_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
      auto dispatch_scatter_add = [](Tensor & self, Dimname dim, const Tensor & index, const Tensor & src) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter_add(dim, index, src);
      };
      return wrap(dispatch_scatter_add(self, _r.dimname(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
      auto dispatch_scatter_add = [](Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter_add(dim, index, src);
      };
      return wrap(dispatch_scatter_add(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// scatter_add_
static PyObject * THPVariable_scatter_add_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "scatter_add_(int64_t dim, Tensor index, Tensor src)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
  auto dispatch_scatter_add_ = [](Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.scatter_add_(dim, index, src);
  };
  return wrap(dispatch_scatter_add_(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// select
static PyObject * THPVariable_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "select(Dimname dim, int64_t index)",
    "select(int64_t dim, int64_t index)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)
      auto dispatch_select = [](Tensor & self, Dimname dim, int64_t index) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.select(dim, index);
      };
      return wrap(dispatch_select(self, _r.dimname(0), _r.toInt64(1)));
    }
    case 1: {
      // aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
      auto dispatch_select = [](Tensor & self, int64_t dim, int64_t index) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.select(dim, index);
      };
      return wrap(dispatch_select(self, _r.toInt64(0), _r.toInt64(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// set_
static PyObject * THPVariable_set_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "set_()",
    "set_(Storage source)",
    "set_(Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride=None)",
    "set_(Tensor source)",
  }, /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::set_(Tensor(a!) self) -> Tensor(a!)
      auto dispatch_set_ = [](Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.set_();
      };
      return wrap(dispatch_set_(self));
    }
    case 1: {
      // aten::set_.source_Storage(Tensor(a!) self, Storage source) -> Tensor(a!)
      auto dispatch_set_ = [](Tensor & self, Storage source) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.set_(source);
      };
      return wrap(dispatch_set_(self, _r.storage(0)));
    }
    case 2: {
      // aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)
      auto dispatch_set_ = [](Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.set_(source, storage_offset, size, stride);
      };
      return wrap(dispatch_set_(self, _r.storage(0), _r.toInt64(1), _r.intlist(2), _r.intlist(3)));
    }
    case 3: {
      // aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)
      auto dispatch_set_ = [](Tensor & self, const Tensor & source) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.set_(source);
      };
      return wrap(dispatch_set_(self, _r.tensor(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sigmoid
static PyObject * THPVariable_sigmoid(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::sigmoid(Tensor self) -> Tensor
  auto dispatch_sigmoid = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sigmoid();
  };
  return wrap(dispatch_sigmoid(self));
  END_HANDLE_TH_ERRORS
}

// sigmoid_
static PyObject * THPVariable_sigmoid_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_sigmoid_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sigmoid_();
  };
  return wrap(dispatch_sigmoid_(self));
  END_HANDLE_TH_ERRORS
}

// sign
static PyObject * THPVariable_sign(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::sign(Tensor self) -> Tensor
  auto dispatch_sign = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sign();
  };
  return wrap(dispatch_sign(self));
  END_HANDLE_TH_ERRORS
}

// sign_
static PyObject * THPVariable_sign_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::sign_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_sign_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sign_();
  };
  return wrap(dispatch_sign_(self));
  END_HANDLE_TH_ERRORS
}

// sin
static PyObject * THPVariable_sin(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::sin(Tensor self) -> Tensor
  auto dispatch_sin = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sin();
  };
  return wrap(dispatch_sin(self));
  END_HANDLE_TH_ERRORS
}

// sin_
static PyObject * THPVariable_sin_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::sin_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_sin_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sin_();
  };
  return wrap(dispatch_sin_(self));
  END_HANDLE_TH_ERRORS
}

// sinh
static PyObject * THPVariable_sinh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::sinh(Tensor self) -> Tensor
  auto dispatch_sinh = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sinh();
  };
  return wrap(dispatch_sinh(self));
  END_HANDLE_TH_ERRORS
}

// sinh_
static PyObject * THPVariable_sinh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::sinh_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_sinh_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sinh_();
  };
  return wrap(dispatch_sinh_(self));
  END_HANDLE_TH_ERRORS
}

// slogdet
static PyObject * THPVariable_slogdet(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"sign", ""}, {"logabsdet", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.slogdet", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
  auto dispatch_slogdet = [](Tensor & self) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.slogdet();
  };
  return wrap(&NamedTuple, dispatch_slogdet(self));
  END_HANDLE_TH_ERRORS
}

// smm
static PyObject * THPVariable_smm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "smm(Tensor mat2)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::smm(Tensor self, Tensor mat2) -> Tensor
  auto dispatch_smm = [](Tensor & self, const Tensor & mat2) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.smm(mat2);
  };
  return wrap(dispatch_smm(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// softmax
static PyObject * THPVariable_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "softmax(Dimname dim, *, ScalarType? dtype=None)",
    "softmax(int64_t dim, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_softmax = [](Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.softmax(dim, dtype);
      };
      return wrap(dispatch_softmax(self, _r.dimname(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      // aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
      auto dispatch_softmax = [](Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.softmax(dim, dtype);
      };
      return wrap(dispatch_softmax(self, _r.toInt64(0), _r.scalartypeOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// solve
static PyObject * THPVariable_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"solution", ""}, {"LU", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.solve", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "solve(Tensor A)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::solve(Tensor self, Tensor A) -> (Tensor solution, Tensor LU)
  auto dispatch_solve = [](Tensor & self, const Tensor & A) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.solve(A);
  };
  return wrap(&NamedTuple, dispatch_solve(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// sort
static PyObject * THPVariable_sort(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.sort", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "sort(Dimname dim, bool descending=False)",
    "sort(int64_t dim=-1, bool descending=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::sort.dimname(Tensor self, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)
      auto dispatch_sort = [](Tensor & self, Dimname dim, bool descending) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.sort(dim, descending);
      };
      return wrap(&NamedTuple, dispatch_sort(self, _r.dimname(0), _r.toBool(1)));
    }
    case 1: {
      // aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
      auto dispatch_sort = [](Tensor & self, int64_t dim, bool descending) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.sort(dim, descending);
      };
      return wrap(&NamedTuple, dispatch_sort(self, _r.toInt64(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sparse_dim
static PyObject * THPVariable_sparse_dim(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::sparse_dim(Tensor self) -> int
  auto dispatch_sparse_dim = [](Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self.sparse_dim();
  };
  return wrap(dispatch_sparse_dim(self));
  END_HANDLE_TH_ERRORS
}

// sparse_mask
static PyObject * THPVariable_sparse_mask(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "sparse_mask(Tensor mask)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::sparse_mask(Tensor self, Tensor mask) -> Tensor
  auto dispatch_sparse_mask = [](Tensor & self, const Tensor & mask) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sparse_mask(mask);
  };
  return wrap(dispatch_sparse_mask(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sparse_resize_
static PyObject * THPVariable_sparse_resize_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "sparse_resize_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
  auto dispatch_sparse_resize_ = [](Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sparse_resize_(size, sparse_dim, dense_dim);
  };
  return wrap(dispatch_sparse_resize_(self, _r.intlist(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sparse_resize_and_clear_
static PyObject * THPVariable_sparse_resize_and_clear_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "sparse_resize_and_clear_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
  auto dispatch_sparse_resize_and_clear_ = [](Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sparse_resize_and_clear_(size, sparse_dim, dense_dim);
  };
  return wrap(dispatch_sparse_resize_and_clear_(self, _r.intlist(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// split
static PyObject * THPVariable_split(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "split(int64_t split_size, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]
  auto dispatch_split = [](Tensor & self, int64_t split_size, int64_t dim) -> std::vector<Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.split(split_size, dim);
  };
  return wrap(dispatch_split(self, _r.toInt64(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// split_with_sizes
static PyObject * THPVariable_split_with_sizes(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "split_with_sizes(IntArrayRef split_sizes, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]
  auto dispatch_split_with_sizes = [](Tensor & self, IntArrayRef split_sizes, int64_t dim) -> std::vector<Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.split_with_sizes(split_sizes, dim);
  };
  return wrap(dispatch_split_with_sizes(self, _r.intlist(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sqrt
static PyObject * THPVariable_sqrt(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::sqrt(Tensor self) -> Tensor
  auto dispatch_sqrt = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sqrt();
  };
  return wrap(dispatch_sqrt(self));
  END_HANDLE_TH_ERRORS
}

// sqrt_
static PyObject * THPVariable_sqrt_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::sqrt_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_sqrt_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sqrt_();
  };
  return wrap(dispatch_sqrt_(self));
  END_HANDLE_TH_ERRORS
}

// square
static PyObject * THPVariable_square(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::square(Tensor self) -> Tensor
  auto dispatch_square = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.square();
  };
  return wrap(dispatch_square(self));
  END_HANDLE_TH_ERRORS
}

// square_
static PyObject * THPVariable_square_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::square_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_square_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.square_();
  };
  return wrap(dispatch_square_(self));
  END_HANDLE_TH_ERRORS
}

\
// squeeze
static PyObject * THPVariable_squeeze(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "squeeze()",
    "squeeze(Dimname dim)",
    "squeeze(int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::squeeze(Tensor(a) self) -> Tensor(a)
      auto dispatch_squeeze = [](Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze();
      };
      return wrap(dispatch_squeeze(self));
    }
    case 1: {
      // aten::squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)
      auto dispatch_squeeze = [](Tensor & self, Dimname dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze(dim);
      };
      return wrap(dispatch_squeeze(self, _r.dimname(0)));
    }
    case 2: {
      // aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
      auto dispatch_squeeze = [](Tensor & self, int64_t dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze(dim);
      };
      return wrap(dispatch_squeeze(self, _r.toInt64(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// squeeze_
static PyObject * THPVariable_squeeze_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "squeeze_()",
    "squeeze_(Dimname dim)",
    "squeeze_(int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::squeeze_(Tensor(a!) self) -> Tensor(a!)
      auto dispatch_squeeze_ = [](Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze_();
      };
      return wrap(dispatch_squeeze_(self));
    }
    case 1: {
      // aten::squeeze_.dimname(Tensor(a!) self, Dimname dim) -> Tensor(a!)
      auto dispatch_squeeze_ = [](Tensor & self, Dimname dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze_(dim);
      };
      return wrap(dispatch_squeeze_(self, _r.dimname(0)));
    }
    case 2: {
      // aten::squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)
      auto dispatch_squeeze_ = [](Tensor & self, int64_t dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze_(dim);
      };
      return wrap(dispatch_squeeze_(self, _r.toInt64(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// sspaddmm
static PyObject * THPVariable_sspaddmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "sspaddmm(Scalar beta, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Scalar beta, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_sspaddmm = [](Scalar beta, Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sspaddmm(mat1, mat2, beta, alpha);
      };
      return wrap(dispatch_sspaddmm(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_sspaddmm = [](Scalar beta, Tensor & self, const Tensor & mat1, const Tensor & mat2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sspaddmm(mat1, mat2, beta, 1);
      };
      return wrap(dispatch_sspaddmm(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_sspaddmm = [](Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sspaddmm(mat1, mat2, beta, alpha);
      };
      return wrap(dispatch_sspaddmm(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// std
static PyObject * THPVariable_std(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "std(DimnameList[1] dim, bool unbiased=True, bool keepdim=False)",
    "std(IntArrayRef[1] dim, bool unbiased=True, bool keepdim=False)",
    "std(bool unbiased=True)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::std.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
      auto dispatch_std = [](Tensor & self, DimnameList dim, bool unbiased, bool keepdim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.std(dim, unbiased, keepdim);
      };
      return wrap(dispatch_std(self, _r.dimnamelist(0), _r.toBool(1), _r.toBool(2)));
    }
    case 1: {
      // aten::std.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
      auto dispatch_std = [](Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.std(dim, unbiased, keepdim);
      };
      return wrap(dispatch_std(self, _r.intlist(0), _r.toBool(1), _r.toBool(2)));
    }
    case 2: {
      // aten::std(Tensor self, bool unbiased=True) -> Tensor
      auto dispatch_std = [](Tensor & self, bool unbiased) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.std(unbiased);
      };
      return wrap(dispatch_std(self, _r.toBool(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// stft
static PyObject * THPVariable_stft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "stft(int64_t n_fft, int64_t? hop_length=None, int64_t? win_length=None, Tensor? window=None, bool normalized=False, bool onesided=True)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool onesided=True) -> Tensor
  auto dispatch_stft = [](Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.stft(n_fft, hop_length, win_length, window, normalized, onesided);
  };
  return wrap(dispatch_stft(self, _r.toInt64(0), _r.toInt64Optional(1), _r.toInt64Optional(2), _r.tensor(3), _r.toBool(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// sub
static PyObject * THPVariable_sub(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "sub(Scalar alpha, Tensor other)|deprecated",
    "sub(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      auto dispatch_sub = [](Tensor & self, Scalar alpha, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sub(other, alpha);
      };
      return wrap(dispatch_sub(self, _r.scalar(0), _r.tensor(1)));
    }
    case 1: {
      // aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      auto dispatch_sub = [](Tensor & self, const Tensor & other, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sub(other, alpha);
      };
      return wrap(dispatch_sub(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// sub_
static PyObject * THPVariable_sub_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "sub_(Scalar alpha, Tensor other)|deprecated",
    "sub_(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_sub_ = [](Tensor & self, Scalar alpha, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sub_(other, alpha);
      };
      return wrap(dispatch_sub_(self, _r.scalar(0), _r.tensor(1)));
    }
    case 1: {
      // aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_sub_ = [](Tensor & self, const Tensor & other, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sub_(other, alpha);
      };
      return wrap(dispatch_sub_(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// sum
static PyObject * THPVariable_sum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "sum(*, ScalarType? dtype=None)",
    "sum(DimnameList[1] dim, bool keepdim=False, *, ScalarType? dtype=None)",
    "sum(IntArrayRef[1] dim, bool keepdim=False, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_sum = [](Tensor & self, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sum(dtype);
      };
      return wrap(dispatch_sum(self, _r.scalartypeOptional(0)));
    }
    case 1: {
      // aten::sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_sum = [](Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sum(dim, keepdim, dtype);
      };
      return wrap(dispatch_sum(self, _r.dimnamelist(0), _r.toBool(1), _r.scalartypeOptional(2)));
    }
    case 2: {
      // aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_sum = [](Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sum(dim, keepdim, dtype);
      };
      return wrap(dispatch_sum(self, _r.intlist(0), _r.toBool(1), _r.scalartypeOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sum_to_size
static PyObject * THPVariable_sum_to_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "sum_to_size(IntArrayRef size)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::sum_to_size(Tensor self, int[] size) -> Tensor
  auto dispatch_sum_to_size = [](Tensor & self, IntArrayRef size) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sum_to_size(size);
  };
  return wrap(dispatch_sum_to_size(self, _r.intlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// svd
static PyObject * THPVariable_svd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"U", ""}, {"S", ""}, {"V", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.svd", nullptr, NamedTuple_fields, 3 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "svd(bool some=True, bool compute_uv=True)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)
  auto dispatch_svd = [](Tensor & self, bool some, bool compute_uv) -> std::tuple<Tensor,Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.svd(some, compute_uv);
  };
  return wrap(&NamedTuple, dispatch_svd(self, _r.toBool(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// symeig
static PyObject * THPVariable_symeig(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"eigenvalues", ""}, {"eigenvectors", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.symeig", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "symeig(bool eigenvectors=False, bool upper=True)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor eigenvalues, Tensor eigenvectors)
  auto dispatch_symeig = [](Tensor & self, bool eigenvectors, bool upper) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.symeig(eigenvectors, upper);
  };
  return wrap(&NamedTuple, dispatch_symeig(self, _r.toBool(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// t
static PyObject * THPVariable_t(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::t(Tensor(a) self) -> Tensor(a)
  auto dispatch_t = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.t();
  };
  return wrap(dispatch_t(self));
  END_HANDLE_TH_ERRORS
}

// t_
static PyObject * THPVariable_t_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::t_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_t_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.t_();
  };
  return wrap(dispatch_t_(self));
  END_HANDLE_TH_ERRORS
}

// take
static PyObject * THPVariable_take(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "take(Tensor index)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::take(Tensor self, Tensor index) -> Tensor
  auto dispatch_take = [](Tensor & self, const Tensor & index) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.take(index);
  };
  return wrap(dispatch_take(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tan
static PyObject * THPVariable_tan(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::tan(Tensor self) -> Tensor
  auto dispatch_tan = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tan();
  };
  return wrap(dispatch_tan(self));
  END_HANDLE_TH_ERRORS
}

// tan_
static PyObject * THPVariable_tan_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::tan_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_tan_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tan_();
  };
  return wrap(dispatch_tan_(self));
  END_HANDLE_TH_ERRORS
}

// tanh
static PyObject * THPVariable_tanh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::tanh(Tensor self) -> Tensor
  auto dispatch_tanh = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tanh();
  };
  return wrap(dispatch_tanh(self));
  END_HANDLE_TH_ERRORS
}

// tanh_
static PyObject * THPVariable_tanh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::tanh_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_tanh_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tanh_();
  };
  return wrap(dispatch_tanh_(self));
  END_HANDLE_TH_ERRORS
}

// to_dense
static PyObject * THPVariable_to_dense(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::to_dense(Tensor self) -> Tensor
  auto dispatch_to_dense = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.to_dense();
  };
  return wrap(dispatch_to_dense(self));
  END_HANDLE_TH_ERRORS
}

// to_mkldnn
static PyObject * THPVariable_to_mkldnn(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::to_mkldnn(Tensor self) -> Tensor
  auto dispatch_to_mkldnn = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.to_mkldnn();
  };
  return wrap(dispatch_to_mkldnn(self));
  END_HANDLE_TH_ERRORS
}

\
// to_sparse
static PyObject * THPVariable_to_sparse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "to_sparse()",
    "to_sparse(int64_t sparse_dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::to_sparse(Tensor self) -> Tensor
      auto dispatch_to_sparse = [](Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.to_sparse();
      };
      return wrap(dispatch_to_sparse(self));
    }
    case 1: {
      // aten::to_sparse.sparse_dim(Tensor self, int sparse_dim) -> Tensor
      auto dispatch_to_sparse = [](Tensor & self, int64_t sparse_dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.to_sparse(sparse_dim);
      };
      return wrap(dispatch_to_sparse(self, _r.toInt64(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// topk
static PyObject * THPVariable_topk(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.topk", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "topk(int64_t k, int64_t dim=-1, bool largest=True, bool sorted=True)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
  auto dispatch_topk = [](Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.topk(k, dim, largest, sorted);
  };
  return wrap(&NamedTuple, dispatch_topk(self, _r.toInt64(0), _r.toInt64(1), _r.toBool(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// trace
static PyObject * THPVariable_trace(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::trace(Tensor self) -> Tensor
  auto dispatch_trace = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.trace();
  };
  return wrap(dispatch_trace(self));
  END_HANDLE_TH_ERRORS
}

\
// transpose
static PyObject * THPVariable_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "transpose(Dimname dim0, Dimname dim1)",
    "transpose(int64_t dim0, int64_t dim1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)
      auto dispatch_transpose = [](Tensor & self, Dimname dim0, Dimname dim1) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.transpose(dim0, dim1);
      };
      return wrap(dispatch_transpose(self, _r.dimname(0), _r.dimname(1)));
    }
    case 1: {
      // aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
      auto dispatch_transpose = [](Tensor & self, int64_t dim0, int64_t dim1) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.transpose(dim0, dim1);
      };
      return wrap(dispatch_transpose(self, _r.toInt64(0), _r.toInt64(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// transpose_
static PyObject * THPVariable_transpose_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "transpose_(int64_t dim0, int64_t dim1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
  auto dispatch_transpose_ = [](Tensor & self, int64_t dim0, int64_t dim1) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.transpose_(dim0, dim1);
  };
  return wrap(dispatch_transpose_(self, _r.toInt64(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// triangular_solve
static PyObject * THPVariable_triangular_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"solution", ""}, {"cloned_coefficient", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.triangular_solve", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "triangular_solve(Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)
  auto dispatch_triangular_solve = [](Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.triangular_solve(A, upper, transpose, unitriangular);
  };
  return wrap(&NamedTuple, dispatch_triangular_solve(self, _r.tensor(0), _r.toBool(1), _r.toBool(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tril
static PyObject * THPVariable_tril(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "tril(int64_t diagonal=0)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::tril(Tensor self, int diagonal=0) -> Tensor
  auto dispatch_tril = [](Tensor & self, int64_t diagonal) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tril(diagonal);
  };
  return wrap(dispatch_tril(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tril_
static PyObject * THPVariable_tril_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "tril_(int64_t diagonal=0)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
  auto dispatch_tril_ = [](Tensor & self, int64_t diagonal) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tril_(diagonal);
  };
  return wrap(dispatch_tril_(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// triu
static PyObject * THPVariable_triu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "triu(int64_t diagonal=0)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::triu(Tensor self, int diagonal=0) -> Tensor
  auto dispatch_triu = [](Tensor & self, int64_t diagonal) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.triu(diagonal);
  };
  return wrap(dispatch_triu(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// triu_
static PyObject * THPVariable_triu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "triu_(int64_t diagonal=0)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
  auto dispatch_triu_ = [](Tensor & self, int64_t diagonal) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.triu_(diagonal);
  };
  return wrap(dispatch_triu_(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// trunc
static PyObject * THPVariable_trunc(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::trunc(Tensor self) -> Tensor
  auto dispatch_trunc = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.trunc();
  };
  return wrap(dispatch_trunc(self));
  END_HANDLE_TH_ERRORS
}

// trunc_
static PyObject * THPVariable_trunc_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::trunc_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_trunc_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.trunc_();
  };
  return wrap(dispatch_trunc_(self));
  END_HANDLE_TH_ERRORS
}

// type_as
static PyObject * THPVariable_type_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "type_as(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::type_as(Tensor self, Tensor other) -> Tensor
  auto dispatch_type_as = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.type_as(other);
  };
  return wrap(dispatch_type_as(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// unbind
static PyObject * THPVariable_unbind(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "unbind(Dimname dim)",
    "unbind(int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::unbind.Dimname(Tensor(a) self, Dimname dim) -> Tensor(a)[]
      auto dispatch_unbind = [](Tensor & self, Dimname dim) -> std::vector<Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.unbind(dim);
      };
      return wrap(dispatch_unbind(self, _r.dimname(0)));
    }
    case 1: {
      // aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]
      auto dispatch_unbind = [](Tensor & self, int64_t dim) -> std::vector<Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.unbind(dim);
      };
      return wrap(dispatch_unbind(self, _r.toInt64(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// unflatten
static PyObject * THPVariable_unflatten(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "unflatten(Dimname dim, IntArrayRef sizes, DimnameList names)",
    "unflatten(int64_t dim, IntArrayRef sizes, DimnameList names)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::unflatten.Dimname(Tensor self, Dimname dim, int[] sizes, Dimname[] names) -> Tensor
      auto dispatch_unflatten = [](Tensor & self, Dimname dim, IntArrayRef sizes, DimnameList names) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.unflatten(dim, sizes, names);
      };
      return wrap(dispatch_unflatten(self, _r.dimname(0), _r.intlist(1), _r.dimnamelist(2)));
    }
    case 1: {
      // aten::unflatten.int(Tensor self, int dim, int[] sizes, Dimname[] names) -> Tensor
      auto dispatch_unflatten = [](Tensor & self, int64_t dim, IntArrayRef sizes, DimnameList names) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.unflatten(dim, sizes, names);
      };
      return wrap(dispatch_unflatten(self, _r.toInt64(0), _r.intlist(1), _r.dimnamelist(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unfold
static PyObject * THPVariable_unfold(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "unfold(int64_t dimension, int64_t size, int64_t step)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)
  auto dispatch_unfold = [](Tensor & self, int64_t dimension, int64_t size, int64_t step) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.unfold(dimension, size, step);
  };
  return wrap(dispatch_unfold(self, _r.toInt64(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// uniform_
static PyObject * THPVariable_uniform_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "uniform_(double from=0, double to=1, *, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)
  auto dispatch_uniform_ = [](Tensor & self, double from, double to, Generator * generator) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.uniform_(from, to, generator);
  };
  return wrap(dispatch_uniform_(self, _r.toDouble(0), _r.toDouble(1), _r.generator(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unsqueeze
static PyObject * THPVariable_unsqueeze(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "unsqueeze(int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
  auto dispatch_unsqueeze = [](Tensor & self, int64_t dim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.unsqueeze(dim);
  };
  return wrap(dispatch_unsqueeze(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unsqueeze_
static PyObject * THPVariable_unsqueeze_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "unsqueeze_(int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)
  auto dispatch_unsqueeze_ = [](Tensor & self, int64_t dim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.unsqueeze_(dim);
  };
  return wrap(dispatch_unsqueeze_(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// values
static PyObject * THPVariable_values(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::values(Tensor(a) self) -> Tensor(a)
  auto dispatch_values = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.values();
  };
  return wrap(dispatch_values(self));
  END_HANDLE_TH_ERRORS
}

\
// var
static PyObject * THPVariable_var(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "var(DimnameList[1] dim, bool unbiased=True, bool keepdim=False)",
    "var(IntArrayRef[1] dim, bool unbiased=True, bool keepdim=False)",
    "var(bool unbiased=True)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::var.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
      auto dispatch_var = [](Tensor & self, DimnameList dim, bool unbiased, bool keepdim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.var(dim, unbiased, keepdim);
      };
      return wrap(dispatch_var(self, _r.dimnamelist(0), _r.toBool(1), _r.toBool(2)));
    }
    case 1: {
      // aten::var.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
      auto dispatch_var = [](Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.var(dim, unbiased, keepdim);
      };
      return wrap(dispatch_var(self, _r.intlist(0), _r.toBool(1), _r.toBool(2)));
    }
    case 2: {
      // aten::var(Tensor self, bool unbiased=True) -> Tensor
      auto dispatch_var = [](Tensor & self, bool unbiased) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.var(unbiased);
      };
      return wrap(dispatch_var(self, _r.toBool(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// view
static PyObject * THPVariable_view(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "view(IntArrayRef size)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::view(Tensor(a) self, int[] size) -> Tensor(a)
  auto dispatch_view = [](Tensor & self, IntArrayRef size) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.view(size);
  };
  return wrap(dispatch_view(self, _r.intlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// view_as
static PyObject * THPVariable_view_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "view_as(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::view_as(Tensor self, Tensor other) -> Tensor
  auto dispatch_view_as = [](Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.view_as(other);
  };
  return wrap(dispatch_view_as(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// where
static PyObject * THPVariable_where(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  static PythonArgParser parser({
    "where(Tensor condition, Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
  auto dispatch_where = [](const Tensor & condition, Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.where(condition, other);
  };
  return wrap(dispatch_where(_r.tensor(0), self, _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// zero_
static PyObject * THPVariable_zero_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  // aten::zero_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_zero_ = [](Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.zero_();
  };
  return wrap(dispatch_zero_(self));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_bool_scalar(PyObject* self, PyObject* args) {
  jit::tracer::warn("Converting a tensor to a Python boolean", jit::tracer::WARN_PYTHON_DATAFLOW);
  return THPVariable_is_nonzero(self, args);
}

// Wrapper converts a raised TypeError into returning NotImplemented
// Used to implement binary arithmetic operators
template <PyObject* (*Func)(PyObject*, PyObject*, PyObject*)>
static PyObject * TypeError_to_NotImplemented_(PyObject* self, PyObject* args, PyObject* kwargs) {
  PyObject* ret = Func(self, args, kwargs);
  if (!ret && PyErr_ExceptionMatches(PyExc_TypeError)) {
    PyErr_Clear();
    Py_INCREF(Py_NotImplemented);
    ret = Py_NotImplemented;
  }
  return ret;
}

// XXX: ops that are bound here are not exposed to the C++ api nor the JIT.
// Any new ops added here should be accompanied with a comment why they are not
// being registered through native_functions.yaml, and be tagged cpp / JIT
PyMethodDef variable_methods[] = {
  // These magic methods are all implemented on python object to wrap NotImplementedError
  {"__add__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_add>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__radd__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_add>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__iadd__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_add_>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__rmul__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_mul>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__mul__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_mul>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__imul__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_mul_>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__sub__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_sub>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__isub__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_sub_>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__div__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_div>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__truediv__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_div>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__idiv__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_div_>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__mod__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_remainder>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__bool__", (PyCFunction)THPVariable_bool_scalar, METH_NOARGS, NULL},
  {"__float__", (PyCFunction)THPVariable_float_scalar, METH_NOARGS, NULL},
  {"__int__", (PyCFunction)THPVariable_integral_scalar, METH_NOARGS, NULL},
  {"__long__", (PyCFunction)THPVariable_integral_scalar, METH_NOARGS, NULL},
  {"__index__", (PyCFunction)THPVariable_index_scalar, METH_NOARGS, NULL},
  {"__nonzero__", (PyCFunction)THPVariable_bool_scalar, METH_NOARGS, NULL},
  {"__invert__", (PyCFunction)THPVariable_invert, METH_NOARGS, NULL},
  {"__matmul__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_matmul>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_is_view", (PyCFunction)THPVariable__is_view, METH_NOARGS, NULL},
  {"apply_", (PyCFunction)THPVariable_apply_, METH_O, NULL},
  {"bfloat16", (PyCFunction)(void(*)(void))THPVariable_bfloat16, METH_VARARGS | METH_KEYWORDS, NULL},
  {"byte", (PyCFunction)(void(*)(void))THPVariable_byte, METH_VARARGS | METH_KEYWORDS, NULL},
  {"char", (PyCFunction)(void(*)(void))THPVariable_char, METH_VARARGS | METH_KEYWORDS, NULL},
  {"contiguous", (PyCFunction)(void(*)(void))THPVariable_contiguous, METH_VARARGS | METH_KEYWORDS, NULL},
  {"copy_", (PyCFunction)(void(*)(void))THPVariable_copy_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cpu", (PyCFunction)(void(*)(void))THPVariable_cpu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cuda", (PyCFunction)(void(*)(void))THPVariable_cuda, METH_VARARGS | METH_KEYWORDS, NULL},
  {"data_ptr", (PyCFunction)THPVariable_data_ptr, METH_NOARGS, NULL},
  {"dim", (PyCFunction)THPVariable_dim, METH_NOARGS, NULL},
  {"has_names", (PyCFunction)THPVariable_has_names, METH_NOARGS, NULL},
  {"double", (PyCFunction)(void(*)(void))THPVariable_double, METH_VARARGS | METH_KEYWORDS, NULL},
  {"element_size", (PyCFunction)THPVariable_element_size, METH_NOARGS, NULL},
  {"float", (PyCFunction)(void(*)(void))THPVariable_float, METH_VARARGS | METH_KEYWORDS, NULL},
  {"get_device", (PyCFunction)THPVariable_get_device, METH_NOARGS, NULL},
  {"bool", (PyCFunction)(void(*)(void))THPVariable_bool, METH_VARARGS | METH_KEYWORDS, NULL},
  {"half", (PyCFunction)(void(*)(void))THPVariable_half, METH_VARARGS | METH_KEYWORDS, NULL},
  {"int", (PyCFunction)(void(*)(void))THPVariable_int, METH_VARARGS | METH_KEYWORDS, NULL},
  {"is_contiguous", (PyCFunction)(void(*)(void))THPVariable_is_contiguous, METH_VARARGS | METH_KEYWORDS, NULL},
  {"item", (PyCFunction)THPVariable_item, METH_NOARGS, NULL},
  {"long", (PyCFunction)(void(*)(void))THPVariable_long, METH_VARARGS | METH_KEYWORDS, NULL},
  {"map_", (PyCFunction)(void(*)(void))THPVariable_map_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"map2_", (PyCFunction)(void(*)(void))THPVariable_map2_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ndimension", (PyCFunction)THPVariable_dim, METH_NOARGS, NULL},
  {"nelement", (PyCFunction)THPVariable_numel, METH_NOARGS, NULL},
  {"new", (PyCFunction)(void(*)(void))THPVariable_new, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_ones", (PyCFunction)(void(*)(void))THPVariable_new_ones, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_tensor", (PyCFunction)(void(*)(void))THPVariable_new_tensor, METH_VARARGS | METH_KEYWORDS, NULL},
  {"nonzero", (PyCFunction)(void(*)(void))THPVariable_nonzero, METH_VARARGS | METH_KEYWORDS, NULL},
  {"numel", (PyCFunction)THPVariable_numel, METH_NOARGS, NULL},
  {"numpy", (PyCFunction)THPVariable_numpy, METH_NOARGS, NULL},
  {"record_stream", (PyCFunction)THPVariable_record_stream, METH_O, NULL},
  {"requires_grad_", (PyCFunction)(void(*)(void))THPVariable_requires_grad_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"short", (PyCFunction)(void(*)(void))THPVariable_short, METH_VARARGS | METH_KEYWORDS, NULL},
  {"size", (PyCFunction)(void(*)(void))THPVariable_size, METH_VARARGS | METH_KEYWORDS, NULL},
  {"storage", (PyCFunction)THPVariable_storage, METH_NOARGS, NULL},
  {"storage_offset", (PyCFunction)THPVariable_storage_offset, METH_NOARGS, NULL},
  {"storage_type", (PyCFunction)THPVariable_storage_type, METH_NOARGS, NULL},
  {"stride", (PyCFunction)(void(*)(void))THPVariable_stride, METH_VARARGS | METH_KEYWORDS, NULL},
  {"to", (PyCFunction)(void(*)(void))THPVariable_to, METH_VARARGS | METH_KEYWORDS, NULL},
  {"tolist", (PyCFunction)THPVariable_tolist, METH_NOARGS, NULL},
  {"type", (PyCFunction)(void(*)(void))THPVariable_type, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__and__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___and__>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__iand__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___iand__>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__ilshift__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___ilshift__>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__ior__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___ior__>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__irshift__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___irshift__>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__ixor__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___ixor__>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__lshift__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___lshift__>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__or__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___or__>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__rshift__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___rshift__>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__xor__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___xor__>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_coalesced_", (PyCFunction)(void(*)(void))THPVariable__coalesced_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_dimI", (PyCFunction)THPVariable__dimI, METH_NOARGS, NULL},
  {"_dimV", (PyCFunction)THPVariable__dimV, METH_NOARGS, NULL},
  {"_indices", (PyCFunction)THPVariable__indices, METH_NOARGS, NULL},
  {"_nnz", (PyCFunction)THPVariable__nnz, METH_NOARGS, NULL},
  {"_values", (PyCFunction)THPVariable__values, METH_NOARGS, NULL},
  {"abs", (PyCFunction)THPVariable_abs, METH_NOARGS, NULL},
  {"abs_", (PyCFunction)THPVariable_abs_, METH_NOARGS, NULL},
  {"acos", (PyCFunction)THPVariable_acos, METH_NOARGS, NULL},
  {"acos_", (PyCFunction)THPVariable_acos_, METH_NOARGS, NULL},
  {"add", (PyCFunction)(void(*)(void))THPVariable_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"add_", (PyCFunction)(void(*)(void))THPVariable_add_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addbmm", (PyCFunction)(void(*)(void))THPVariable_addbmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addbmm_", (PyCFunction)(void(*)(void))THPVariable_addbmm_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addcdiv", (PyCFunction)(void(*)(void))THPVariable_addcdiv, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addcdiv_", (PyCFunction)(void(*)(void))THPVariable_addcdiv_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addcmul", (PyCFunction)(void(*)(void))THPVariable_addcmul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addcmul_", (PyCFunction)(void(*)(void))THPVariable_addcmul_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addmm", (PyCFunction)(void(*)(void))THPVariable_addmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addmm_", (PyCFunction)(void(*)(void))THPVariable_addmm_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addmv", (PyCFunction)(void(*)(void))THPVariable_addmv, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addmv_", (PyCFunction)(void(*)(void))THPVariable_addmv_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addr", (PyCFunction)(void(*)(void))THPVariable_addr, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addr_", (PyCFunction)(void(*)(void))THPVariable_addr_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"align_as", (PyCFunction)(void(*)(void))THPVariable_align_as, METH_VARARGS | METH_KEYWORDS, NULL},
  {"align_to", (PyCFunction)(void(*)(void))THPVariable_align_to, METH_VARARGS | METH_KEYWORDS, NULL},
  {"all", (PyCFunction)(void(*)(void))THPVariable_all, METH_VARARGS | METH_KEYWORDS, NULL},
  {"allclose", (PyCFunction)(void(*)(void))THPVariable_allclose, METH_VARARGS | METH_KEYWORDS, NULL},
  {"angle", (PyCFunction)THPVariable_angle, METH_NOARGS, NULL},
  {"any", (PyCFunction)(void(*)(void))THPVariable_any, METH_VARARGS | METH_KEYWORDS, NULL},
  {"argmax", (PyCFunction)(void(*)(void))THPVariable_argmax, METH_VARARGS | METH_KEYWORDS, NULL},
  {"argmin", (PyCFunction)(void(*)(void))THPVariable_argmin, METH_VARARGS | METH_KEYWORDS, NULL},
  {"argsort", (PyCFunction)(void(*)(void))THPVariable_argsort, METH_VARARGS | METH_KEYWORDS, NULL},
  {"as_strided", (PyCFunction)(void(*)(void))THPVariable_as_strided, METH_VARARGS | METH_KEYWORDS, NULL},
  {"as_strided_", (PyCFunction)(void(*)(void))THPVariable_as_strided_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"asin", (PyCFunction)THPVariable_asin, METH_NOARGS, NULL},
  {"asin_", (PyCFunction)THPVariable_asin_, METH_NOARGS, NULL},
  {"atan", (PyCFunction)THPVariable_atan, METH_NOARGS, NULL},
  {"atan2", (PyCFunction)(void(*)(void))THPVariable_atan2, METH_VARARGS | METH_KEYWORDS, NULL},
  {"atan2_", (PyCFunction)(void(*)(void))THPVariable_atan2_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"atan_", (PyCFunction)THPVariable_atan_, METH_NOARGS, NULL},
  {"backward", (PyCFunction)(void(*)(void))THPVariable_backward, METH_VARARGS | METH_KEYWORDS, NULL},
  {"baddbmm", (PyCFunction)(void(*)(void))THPVariable_baddbmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"baddbmm_", (PyCFunction)(void(*)(void))THPVariable_baddbmm_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bernoulli", (PyCFunction)(void(*)(void))THPVariable_bernoulli, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bernoulli_", (PyCFunction)(void(*)(void))THPVariable_bernoulli_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bincount", (PyCFunction)(void(*)(void))THPVariable_bincount, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bitwise_and", (PyCFunction)(void(*)(void))THPVariable_bitwise_and, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bitwise_and_", (PyCFunction)(void(*)(void))THPVariable_bitwise_and_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bitwise_not", (PyCFunction)THPVariable_bitwise_not, METH_NOARGS, NULL},
  {"bitwise_not_", (PyCFunction)THPVariable_bitwise_not_, METH_NOARGS, NULL},
  {"bitwise_or", (PyCFunction)(void(*)(void))THPVariable_bitwise_or, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bitwise_or_", (PyCFunction)(void(*)(void))THPVariable_bitwise_or_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bitwise_xor", (PyCFunction)(void(*)(void))THPVariable_bitwise_xor, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bitwise_xor_", (PyCFunction)(void(*)(void))THPVariable_bitwise_xor_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bmm", (PyCFunction)(void(*)(void))THPVariable_bmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cauchy_", (PyCFunction)(void(*)(void))THPVariable_cauchy_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ceil", (PyCFunction)THPVariable_ceil, METH_NOARGS, NULL},
  {"ceil_", (PyCFunction)THPVariable_ceil_, METH_NOARGS, NULL},
  {"cholesky", (PyCFunction)(void(*)(void))THPVariable_cholesky, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cholesky_inverse", (PyCFunction)(void(*)(void))THPVariable_cholesky_inverse, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cholesky_solve", (PyCFunction)(void(*)(void))THPVariable_cholesky_solve, METH_VARARGS | METH_KEYWORDS, NULL},
  {"chunk", (PyCFunction)(void(*)(void))THPVariable_chunk, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp", (PyCFunction)(void(*)(void))THPVariable_clamp, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp_", (PyCFunction)(void(*)(void))THPVariable_clamp_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp_max", (PyCFunction)(void(*)(void))THPVariable_clamp_max, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp_max_", (PyCFunction)(void(*)(void))THPVariable_clamp_max_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp_min", (PyCFunction)(void(*)(void))THPVariable_clamp_min, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp_min_", (PyCFunction)(void(*)(void))THPVariable_clamp_min_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clone", (PyCFunction)(void(*)(void))THPVariable_clone, METH_VARARGS | METH_KEYWORDS, NULL},
  {"coalesce", (PyCFunction)THPVariable_coalesce, METH_NOARGS, NULL},
  {"conj", (PyCFunction)THPVariable_conj, METH_NOARGS, NULL},
  {"cos", (PyCFunction)THPVariable_cos, METH_NOARGS, NULL},
  {"cos_", (PyCFunction)THPVariable_cos_, METH_NOARGS, NULL},
  {"cosh", (PyCFunction)THPVariable_cosh, METH_NOARGS, NULL},
  {"cosh_", (PyCFunction)THPVariable_cosh_, METH_NOARGS, NULL},
  {"cross", (PyCFunction)(void(*)(void))THPVariable_cross, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cummax", (PyCFunction)(void(*)(void))THPVariable_cummax, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cummin", (PyCFunction)(void(*)(void))THPVariable_cummin, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cumprod", (PyCFunction)(void(*)(void))THPVariable_cumprod, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cumsum", (PyCFunction)(void(*)(void))THPVariable_cumsum, METH_VARARGS | METH_KEYWORDS, NULL},
  {"dense_dim", (PyCFunction)THPVariable_dense_dim, METH_NOARGS, NULL},
  {"dequantize", (PyCFunction)THPVariable_dequantize, METH_NOARGS, NULL},
  {"det", (PyCFunction)THPVariable_det, METH_NOARGS, NULL},
  {"detach", (PyCFunction)THPVariable_detach, METH_NOARGS, NULL},
  {"detach_", (PyCFunction)THPVariable_detach_, METH_NOARGS, NULL},
  {"diag", (PyCFunction)(void(*)(void))THPVariable_diag, METH_VARARGS | METH_KEYWORDS, NULL},
  {"diag_embed", (PyCFunction)(void(*)(void))THPVariable_diag_embed, METH_VARARGS | METH_KEYWORDS, NULL},
  {"diagflat", (PyCFunction)(void(*)(void))THPVariable_diagflat, METH_VARARGS | METH_KEYWORDS, NULL},
  {"diagonal", (PyCFunction)(void(*)(void))THPVariable_diagonal, METH_VARARGS | METH_KEYWORDS, NULL},
  {"digamma", (PyCFunction)THPVariable_digamma, METH_NOARGS, NULL},
  {"digamma_", (PyCFunction)THPVariable_digamma_, METH_NOARGS, NULL},
  {"dist", (PyCFunction)(void(*)(void))THPVariable_dist, METH_VARARGS | METH_KEYWORDS, NULL},
  {"div", (PyCFunction)(void(*)(void))THPVariable_div, METH_VARARGS | METH_KEYWORDS, NULL},
  {"div_", (PyCFunction)(void(*)(void))THPVariable_div_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"dot", (PyCFunction)(void(*)(void))THPVariable_dot, METH_VARARGS | METH_KEYWORDS, NULL},
  {"eig", (PyCFunction)(void(*)(void))THPVariable_eig, METH_VARARGS | METH_KEYWORDS, NULL},
  {"eq", (PyCFunction)(void(*)(void))THPVariable_eq, METH_VARARGS | METH_KEYWORDS, NULL},
  {"eq_", (PyCFunction)(void(*)(void))THPVariable_eq_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"equal", (PyCFunction)(void(*)(void))THPVariable_equal, METH_VARARGS | METH_KEYWORDS, NULL},
  {"erf", (PyCFunction)THPVariable_erf, METH_NOARGS, NULL},
  {"erf_", (PyCFunction)THPVariable_erf_, METH_NOARGS, NULL},
  {"erfc", (PyCFunction)THPVariable_erfc, METH_NOARGS, NULL},
  {"erfc_", (PyCFunction)THPVariable_erfc_, METH_NOARGS, NULL},
  {"erfinv", (PyCFunction)THPVariable_erfinv, METH_NOARGS, NULL},
  {"erfinv_", (PyCFunction)THPVariable_erfinv_, METH_NOARGS, NULL},
  {"exp", (PyCFunction)THPVariable_exp, METH_NOARGS, NULL},
  {"exp_", (PyCFunction)THPVariable_exp_, METH_NOARGS, NULL},
  {"expand", (PyCFunction)(void(*)(void))THPVariable_expand, METH_VARARGS | METH_KEYWORDS, NULL},
  {"expand_as", (PyCFunction)(void(*)(void))THPVariable_expand_as, METH_VARARGS | METH_KEYWORDS, NULL},
  {"expm1", (PyCFunction)THPVariable_expm1, METH_NOARGS, NULL},
  {"expm1_", (PyCFunction)THPVariable_expm1_, METH_NOARGS, NULL},
  {"exponential_", (PyCFunction)(void(*)(void))THPVariable_exponential_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"fft", (PyCFunction)(void(*)(void))THPVariable_fft, METH_VARARGS | METH_KEYWORDS, NULL},
  {"fill_", (PyCFunction)(void(*)(void))THPVariable_fill_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"fill_diagonal_", (PyCFunction)(void(*)(void))THPVariable_fill_diagonal_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"flatten", (PyCFunction)(void(*)(void))THPVariable_flatten, METH_VARARGS | METH_KEYWORDS, NULL},
  {"flip", (PyCFunction)(void(*)(void))THPVariable_flip, METH_VARARGS | METH_KEYWORDS, NULL},
  {"floor", (PyCFunction)THPVariable_floor, METH_NOARGS, NULL},
  {"floor_", (PyCFunction)THPVariable_floor_, METH_NOARGS, NULL},
  {"fmod", (PyCFunction)(void(*)(void))THPVariable_fmod, METH_VARARGS | METH_KEYWORDS, NULL},
  {"fmod_", (PyCFunction)(void(*)(void))THPVariable_fmod_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"frac", (PyCFunction)THPVariable_frac, METH_NOARGS, NULL},
  {"frac_", (PyCFunction)THPVariable_frac_, METH_NOARGS, NULL},
  {"gather", (PyCFunction)(void(*)(void))THPVariable_gather, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ge", (PyCFunction)(void(*)(void))THPVariable_ge, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ge_", (PyCFunction)(void(*)(void))THPVariable_ge_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"geometric_", (PyCFunction)(void(*)(void))THPVariable_geometric_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"geqrf", (PyCFunction)THPVariable_geqrf, METH_NOARGS, NULL},
  {"ger", (PyCFunction)(void(*)(void))THPVariable_ger, METH_VARARGS | METH_KEYWORDS, NULL},
  {"gt", (PyCFunction)(void(*)(void))THPVariable_gt, METH_VARARGS | METH_KEYWORDS, NULL},
  {"gt_", (PyCFunction)(void(*)(void))THPVariable_gt_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"hardshrink", (PyCFunction)(void(*)(void))THPVariable_hardshrink, METH_VARARGS | METH_KEYWORDS, NULL},
  {"histc", (PyCFunction)(void(*)(void))THPVariable_histc, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ifft", (PyCFunction)(void(*)(void))THPVariable_ifft, METH_VARARGS | METH_KEYWORDS, NULL},
  {"imag", (PyCFunction)THPVariable_imag, METH_NOARGS, NULL},
  {"index_add", (PyCFunction)(void(*)(void))THPVariable_index_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_add_", (PyCFunction)(void(*)(void))THPVariable_index_add_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_copy", (PyCFunction)(void(*)(void))THPVariable_index_copy, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_copy_", (PyCFunction)(void(*)(void))THPVariable_index_copy_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_fill", (PyCFunction)(void(*)(void))THPVariable_index_fill, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_fill_", (PyCFunction)(void(*)(void))THPVariable_index_fill_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_put", (PyCFunction)(void(*)(void))THPVariable_index_put, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_put_", (PyCFunction)(void(*)(void))THPVariable_index_put_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_select", (PyCFunction)(void(*)(void))THPVariable_index_select, METH_VARARGS | METH_KEYWORDS, NULL},
  {"indices", (PyCFunction)THPVariable_indices, METH_NOARGS, NULL},
  {"int_repr", (PyCFunction)THPVariable_int_repr, METH_NOARGS, NULL},
  {"inverse", (PyCFunction)THPVariable_inverse, METH_NOARGS, NULL},
  {"irfft", (PyCFunction)(void(*)(void))THPVariable_irfft, METH_VARARGS | METH_KEYWORDS, NULL},
  {"is_coalesced", (PyCFunction)THPVariable_is_coalesced, METH_NOARGS, NULL},
  {"is_complex", (PyCFunction)THPVariable_is_complex, METH_NOARGS, NULL},
  {"is_distributed", (PyCFunction)THPVariable_is_distributed, METH_NOARGS, NULL},
  {"is_floating_point", (PyCFunction)THPVariable_is_floating_point, METH_NOARGS, NULL},
  {"is_nonzero", (PyCFunction)THPVariable_is_nonzero, METH_NOARGS, NULL},
  {"is_pinned", (PyCFunction)THPVariable_is_pinned, METH_NOARGS, NULL},
  {"is_same_size", (PyCFunction)(void(*)(void))THPVariable_is_same_size, METH_VARARGS | METH_KEYWORDS, NULL},
  {"is_set_to", (PyCFunction)(void(*)(void))THPVariable_is_set_to, METH_VARARGS | METH_KEYWORDS, NULL},
  {"is_signed", (PyCFunction)THPVariable_is_signed, METH_NOARGS, NULL},
  {"isclose", (PyCFunction)(void(*)(void))THPVariable_isclose, METH_VARARGS | METH_KEYWORDS, NULL},
  {"kthvalue", (PyCFunction)(void(*)(void))THPVariable_kthvalue, METH_VARARGS | METH_KEYWORDS, NULL},
  {"le", (PyCFunction)(void(*)(void))THPVariable_le, METH_VARARGS | METH_KEYWORDS, NULL},
  {"le_", (PyCFunction)(void(*)(void))THPVariable_le_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lerp", (PyCFunction)(void(*)(void))THPVariable_lerp, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lerp_", (PyCFunction)(void(*)(void))THPVariable_lerp_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lgamma", (PyCFunction)THPVariable_lgamma, METH_NOARGS, NULL},
  {"lgamma_", (PyCFunction)THPVariable_lgamma_, METH_NOARGS, NULL},
  {"log", (PyCFunction)THPVariable_log, METH_NOARGS, NULL},
  {"log10", (PyCFunction)THPVariable_log10, METH_NOARGS, NULL},
  {"log10_", (PyCFunction)THPVariable_log10_, METH_NOARGS, NULL},
  {"log1p", (PyCFunction)THPVariable_log1p, METH_NOARGS, NULL},
  {"log1p_", (PyCFunction)THPVariable_log1p_, METH_NOARGS, NULL},
  {"log2", (PyCFunction)THPVariable_log2, METH_NOARGS, NULL},
  {"log2_", (PyCFunction)THPVariable_log2_, METH_NOARGS, NULL},
  {"log_", (PyCFunction)THPVariable_log_, METH_NOARGS, NULL},
  {"log_normal_", (PyCFunction)(void(*)(void))THPVariable_log_normal_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"log_softmax", (PyCFunction)(void(*)(void))THPVariable_log_softmax, METH_VARARGS | METH_KEYWORDS, NULL},
  {"logdet", (PyCFunction)THPVariable_logdet, METH_NOARGS, NULL},
  {"logical_and", (PyCFunction)(void(*)(void))THPVariable_logical_and, METH_VARARGS | METH_KEYWORDS, NULL},
  {"logical_and_", (PyCFunction)(void(*)(void))THPVariable_logical_and_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"logical_not", (PyCFunction)THPVariable_logical_not, METH_NOARGS, NULL},
  {"logical_not_", (PyCFunction)THPVariable_logical_not_, METH_NOARGS, NULL},
  {"logical_or", (PyCFunction)(void(*)(void))THPVariable_logical_or, METH_VARARGS | METH_KEYWORDS, NULL},
  {"logical_or_", (PyCFunction)(void(*)(void))THPVariable_logical_or_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"logical_xor", (PyCFunction)(void(*)(void))THPVariable_logical_xor, METH_VARARGS | METH_KEYWORDS, NULL},
  {"logical_xor_", (PyCFunction)(void(*)(void))THPVariable_logical_xor_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"logsumexp", (PyCFunction)(void(*)(void))THPVariable_logsumexp, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lstsq", (PyCFunction)(void(*)(void))THPVariable_lstsq, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lt", (PyCFunction)(void(*)(void))THPVariable_lt, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lt_", (PyCFunction)(void(*)(void))THPVariable_lt_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lu_solve", (PyCFunction)(void(*)(void))THPVariable_lu_solve, METH_VARARGS | METH_KEYWORDS, NULL},
  {"masked_fill", (PyCFunction)(void(*)(void))THPVariable_masked_fill, METH_VARARGS | METH_KEYWORDS, NULL},
  {"masked_fill_", (PyCFunction)(void(*)(void))THPVariable_masked_fill_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"masked_scatter", (PyCFunction)(void(*)(void))THPVariable_masked_scatter, METH_VARARGS | METH_KEYWORDS, NULL},
  {"masked_scatter_", (PyCFunction)(void(*)(void))THPVariable_masked_scatter_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"masked_select", (PyCFunction)(void(*)(void))THPVariable_masked_select, METH_VARARGS | METH_KEYWORDS, NULL},
  {"matmul", (PyCFunction)(void(*)(void))THPVariable_matmul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"matrix_power", (PyCFunction)(void(*)(void))THPVariable_matrix_power, METH_VARARGS | METH_KEYWORDS, NULL},
  {"max", (PyCFunction)(void(*)(void))THPVariable_max, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mean", (PyCFunction)(void(*)(void))THPVariable_mean, METH_VARARGS | METH_KEYWORDS, NULL},
  {"median", (PyCFunction)(void(*)(void))THPVariable_median, METH_VARARGS | METH_KEYWORDS, NULL},
  {"min", (PyCFunction)(void(*)(void))THPVariable_min, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mm", (PyCFunction)(void(*)(void))THPVariable_mm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mode", (PyCFunction)(void(*)(void))THPVariable_mode, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mul", (PyCFunction)(void(*)(void))THPVariable_mul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mul_", (PyCFunction)(void(*)(void))THPVariable_mul_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"multinomial", (PyCFunction)(void(*)(void))THPVariable_multinomial, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mv", (PyCFunction)(void(*)(void))THPVariable_mv, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mvlgamma", (PyCFunction)(void(*)(void))THPVariable_mvlgamma, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mvlgamma_", (PyCFunction)(void(*)(void))THPVariable_mvlgamma_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"narrow", (PyCFunction)(void(*)(void))THPVariable_narrow, METH_VARARGS | METH_KEYWORDS, NULL},
  {"narrow_copy", (PyCFunction)(void(*)(void))THPVariable_narrow_copy, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ne", (PyCFunction)(void(*)(void))THPVariable_ne, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ne_", (PyCFunction)(void(*)(void))THPVariable_ne_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"neg", (PyCFunction)THPVariable_neg, METH_NOARGS, NULL},
  {"neg_", (PyCFunction)THPVariable_neg_, METH_NOARGS, NULL},
  {"new_empty", (PyCFunction)(void(*)(void))THPVariable_new_empty, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_full", (PyCFunction)(void(*)(void))THPVariable_new_full, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_zeros", (PyCFunction)(void(*)(void))THPVariable_new_zeros, METH_VARARGS | METH_KEYWORDS, NULL},
  {"norm", (PyCFunction)(void(*)(void))THPVariable_norm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"normal_", (PyCFunction)(void(*)(void))THPVariable_normal_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"orgqr", (PyCFunction)(void(*)(void))THPVariable_orgqr, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ormqr", (PyCFunction)(void(*)(void))THPVariable_ormqr, METH_VARARGS | METH_KEYWORDS, NULL},
  {"permute", (PyCFunction)(void(*)(void))THPVariable_permute, METH_VARARGS | METH_KEYWORDS, NULL},
  {"pin_memory", (PyCFunction)THPVariable_pin_memory, METH_NOARGS, NULL},
  {"pinverse", (PyCFunction)(void(*)(void))THPVariable_pinverse, METH_VARARGS | METH_KEYWORDS, NULL},
  {"polygamma", (PyCFunction)(void(*)(void))THPVariable_polygamma, METH_VARARGS | METH_KEYWORDS, NULL},
  {"polygamma_", (PyCFunction)(void(*)(void))THPVariable_polygamma_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"pow", (PyCFunction)(void(*)(void))THPVariable_pow, METH_VARARGS | METH_KEYWORDS, NULL},
  {"pow_", (PyCFunction)(void(*)(void))THPVariable_pow_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"prelu", (PyCFunction)(void(*)(void))THPVariable_prelu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"prod", (PyCFunction)(void(*)(void))THPVariable_prod, METH_VARARGS | METH_KEYWORDS, NULL},
  {"put_", (PyCFunction)(void(*)(void))THPVariable_put_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"q_per_channel_axis", (PyCFunction)THPVariable_q_per_channel_axis, METH_NOARGS, NULL},
  {"q_per_channel_scales", (PyCFunction)THPVariable_q_per_channel_scales, METH_NOARGS, NULL},
  {"q_per_channel_zero_points", (PyCFunction)THPVariable_q_per_channel_zero_points, METH_NOARGS, NULL},
  {"q_scale", (PyCFunction)THPVariable_q_scale, METH_NOARGS, NULL},
  {"q_zero_point", (PyCFunction)THPVariable_q_zero_point, METH_NOARGS, NULL},
  {"qr", (PyCFunction)(void(*)(void))THPVariable_qr, METH_VARARGS | METH_KEYWORDS, NULL},
  {"qscheme", (PyCFunction)THPVariable_qscheme, METH_NOARGS, NULL},
  {"random_", (PyCFunction)(void(*)(void))THPVariable_random_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"real", (PyCFunction)THPVariable_real, METH_NOARGS, NULL},
  {"reciprocal", (PyCFunction)THPVariable_reciprocal, METH_NOARGS, NULL},
  {"reciprocal_", (PyCFunction)THPVariable_reciprocal_, METH_NOARGS, NULL},
  {"refine_names", (PyCFunction)(void(*)(void))THPVariable_refine_names, METH_VARARGS | METH_KEYWORDS, NULL},
  {"relu", (PyCFunction)THPVariable_relu, METH_NOARGS, NULL},
  {"relu_", (PyCFunction)THPVariable_relu_, METH_NOARGS, NULL},
  {"remainder", (PyCFunction)(void(*)(void))THPVariable_remainder, METH_VARARGS | METH_KEYWORDS, NULL},
  {"remainder_", (PyCFunction)(void(*)(void))THPVariable_remainder_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"rename", (PyCFunction)(void(*)(void))THPVariable_rename, METH_VARARGS | METH_KEYWORDS, NULL},
  {"rename_", (PyCFunction)(void(*)(void))THPVariable_rename_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"renorm", (PyCFunction)(void(*)(void))THPVariable_renorm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"renorm_", (PyCFunction)(void(*)(void))THPVariable_renorm_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"repeat", (PyCFunction)(void(*)(void))THPVariable_repeat, METH_VARARGS | METH_KEYWORDS, NULL},
  {"repeat_interleave", (PyCFunction)(void(*)(void))THPVariable_repeat_interleave, METH_VARARGS | METH_KEYWORDS, NULL},
  {"reshape", (PyCFunction)(void(*)(void))THPVariable_reshape, METH_VARARGS | METH_KEYWORDS, NULL},
  {"reshape_as", (PyCFunction)(void(*)(void))THPVariable_reshape_as, METH_VARARGS | METH_KEYWORDS, NULL},
  {"resize_", (PyCFunction)(void(*)(void))THPVariable_resize_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"resize_as_", (PyCFunction)(void(*)(void))THPVariable_resize_as_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"rfft", (PyCFunction)(void(*)(void))THPVariable_rfft, METH_VARARGS | METH_KEYWORDS, NULL},
  {"roll", (PyCFunction)(void(*)(void))THPVariable_roll, METH_VARARGS | METH_KEYWORDS, NULL},
  {"rot90", (PyCFunction)(void(*)(void))THPVariable_rot90, METH_VARARGS | METH_KEYWORDS, NULL},
  {"round", (PyCFunction)THPVariable_round, METH_NOARGS, NULL},
  {"round_", (PyCFunction)THPVariable_round_, METH_NOARGS, NULL},
  {"rsqrt", (PyCFunction)THPVariable_rsqrt, METH_NOARGS, NULL},
  {"rsqrt_", (PyCFunction)THPVariable_rsqrt_, METH_NOARGS, NULL},
  {"scatter", (PyCFunction)(void(*)(void))THPVariable_scatter, METH_VARARGS | METH_KEYWORDS, NULL},
  {"scatter_", (PyCFunction)(void(*)(void))THPVariable_scatter_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"scatter_add", (PyCFunction)(void(*)(void))THPVariable_scatter_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"scatter_add_", (PyCFunction)(void(*)(void))THPVariable_scatter_add_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"select", (PyCFunction)(void(*)(void))THPVariable_select, METH_VARARGS | METH_KEYWORDS, NULL},
  {"set_", (PyCFunction)(void(*)(void))THPVariable_set_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sigmoid", (PyCFunction)THPVariable_sigmoid, METH_NOARGS, NULL},
  {"sigmoid_", (PyCFunction)THPVariable_sigmoid_, METH_NOARGS, NULL},
  {"sign", (PyCFunction)THPVariable_sign, METH_NOARGS, NULL},
  {"sign_", (PyCFunction)THPVariable_sign_, METH_NOARGS, NULL},
  {"sin", (PyCFunction)THPVariable_sin, METH_NOARGS, NULL},
  {"sin_", (PyCFunction)THPVariable_sin_, METH_NOARGS, NULL},
  {"sinh", (PyCFunction)THPVariable_sinh, METH_NOARGS, NULL},
  {"sinh_", (PyCFunction)THPVariable_sinh_, METH_NOARGS, NULL},
  {"slogdet", (PyCFunction)THPVariable_slogdet, METH_NOARGS, NULL},
  {"smm", (PyCFunction)(void(*)(void))THPVariable_smm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"softmax", (PyCFunction)(void(*)(void))THPVariable_softmax, METH_VARARGS | METH_KEYWORDS, NULL},
  {"solve", (PyCFunction)(void(*)(void))THPVariable_solve, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sort", (PyCFunction)(void(*)(void))THPVariable_sort, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sparse_dim", (PyCFunction)THPVariable_sparse_dim, METH_NOARGS, NULL},
  {"sparse_mask", (PyCFunction)(void(*)(void))THPVariable_sparse_mask, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sparse_resize_", (PyCFunction)(void(*)(void))THPVariable_sparse_resize_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sparse_resize_and_clear_", (PyCFunction)(void(*)(void))THPVariable_sparse_resize_and_clear_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"split", (PyCFunction)(void(*)(void))THPVariable_split, METH_VARARGS | METH_KEYWORDS, NULL},
  {"split_with_sizes", (PyCFunction)(void(*)(void))THPVariable_split_with_sizes, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sqrt", (PyCFunction)THPVariable_sqrt, METH_NOARGS, NULL},
  {"sqrt_", (PyCFunction)THPVariable_sqrt_, METH_NOARGS, NULL},
  {"square", (PyCFunction)THPVariable_square, METH_NOARGS, NULL},
  {"square_", (PyCFunction)THPVariable_square_, METH_NOARGS, NULL},
  {"squeeze", (PyCFunction)(void(*)(void))THPVariable_squeeze, METH_VARARGS | METH_KEYWORDS, NULL},
  {"squeeze_", (PyCFunction)(void(*)(void))THPVariable_squeeze_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sspaddmm", (PyCFunction)(void(*)(void))THPVariable_sspaddmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"std", (PyCFunction)(void(*)(void))THPVariable_std, METH_VARARGS | METH_KEYWORDS, NULL},
  {"stft", (PyCFunction)(void(*)(void))THPVariable_stft, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sub", (PyCFunction)(void(*)(void))THPVariable_sub, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sub_", (PyCFunction)(void(*)(void))THPVariable_sub_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sum", (PyCFunction)(void(*)(void))THPVariable_sum, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sum_to_size", (PyCFunction)(void(*)(void))THPVariable_sum_to_size, METH_VARARGS | METH_KEYWORDS, NULL},
  {"svd", (PyCFunction)(void(*)(void))THPVariable_svd, METH_VARARGS | METH_KEYWORDS, NULL},
  {"symeig", (PyCFunction)(void(*)(void))THPVariable_symeig, METH_VARARGS | METH_KEYWORDS, NULL},
  {"t", (PyCFunction)THPVariable_t, METH_NOARGS, NULL},
  {"t_", (PyCFunction)THPVariable_t_, METH_NOARGS, NULL},
  {"take", (PyCFunction)(void(*)(void))THPVariable_take, METH_VARARGS | METH_KEYWORDS, NULL},
  {"tan", (PyCFunction)THPVariable_tan, METH_NOARGS, NULL},
  {"tan_", (PyCFunction)THPVariable_tan_, METH_NOARGS, NULL},
  {"tanh", (PyCFunction)THPVariable_tanh, METH_NOARGS, NULL},
  {"tanh_", (PyCFunction)THPVariable_tanh_, METH_NOARGS, NULL},
  {"to_dense", (PyCFunction)THPVariable_to_dense, METH_NOARGS, NULL},
  {"to_mkldnn", (PyCFunction)THPVariable_to_mkldnn, METH_NOARGS, NULL},
  {"to_sparse", (PyCFunction)(void(*)(void))THPVariable_to_sparse, METH_VARARGS | METH_KEYWORDS, NULL},
  {"topk", (PyCFunction)(void(*)(void))THPVariable_topk, METH_VARARGS | METH_KEYWORDS, NULL},
  {"trace", (PyCFunction)THPVariable_trace, METH_NOARGS, NULL},
  {"transpose", (PyCFunction)(void(*)(void))THPVariable_transpose, METH_VARARGS | METH_KEYWORDS, NULL},
  {"transpose_", (PyCFunction)(void(*)(void))THPVariable_transpose_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"triangular_solve", (PyCFunction)(void(*)(void))THPVariable_triangular_solve, METH_VARARGS | METH_KEYWORDS, NULL},
  {"tril", (PyCFunction)(void(*)(void))THPVariable_tril, METH_VARARGS | METH_KEYWORDS, NULL},
  {"tril_", (PyCFunction)(void(*)(void))THPVariable_tril_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"triu", (PyCFunction)(void(*)(void))THPVariable_triu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"triu_", (PyCFunction)(void(*)(void))THPVariable_triu_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"trunc", (PyCFunction)THPVariable_trunc, METH_NOARGS, NULL},
  {"trunc_", (PyCFunction)THPVariable_trunc_, METH_NOARGS, NULL},
  {"type_as", (PyCFunction)(void(*)(void))THPVariable_type_as, METH_VARARGS | METH_KEYWORDS, NULL},
  {"unbind", (PyCFunction)(void(*)(void))THPVariable_unbind, METH_VARARGS | METH_KEYWORDS, NULL},
  {"unflatten", (PyCFunction)(void(*)(void))THPVariable_unflatten, METH_VARARGS | METH_KEYWORDS, NULL},
  {"unfold", (PyCFunction)(void(*)(void))THPVariable_unfold, METH_VARARGS | METH_KEYWORDS, NULL},
  {"uniform_", (PyCFunction)(void(*)(void))THPVariable_uniform_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"unsqueeze", (PyCFunction)(void(*)(void))THPVariable_unsqueeze, METH_VARARGS | METH_KEYWORDS, NULL},
  {"unsqueeze_", (PyCFunction)(void(*)(void))THPVariable_unsqueeze_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"values", (PyCFunction)THPVariable_values, METH_NOARGS, NULL},
  {"var", (PyCFunction)(void(*)(void))THPVariable_var, METH_VARARGS | METH_KEYWORDS, NULL},
  {"view", (PyCFunction)(void(*)(void))THPVariable_view, METH_VARARGS | METH_KEYWORDS, NULL},
  {"view_as", (PyCFunction)(void(*)(void))THPVariable_view_as, METH_VARARGS | METH_KEYWORDS, NULL},
  {"where", (PyCFunction)(void(*)(void))THPVariable_where, METH_VARARGS | METH_KEYWORDS, NULL},
  {"zero_", (PyCFunction)THPVariable_zero_, METH_NOARGS, NULL},
  {NULL}
};

}} // namespace torch::autograd
