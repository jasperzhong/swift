// @generated from tools/autograd/templates/python_torch_functions.cpp

// Python bindings for torch.* functions implemented through ATen.
//
// The functions are bound as static methods on a class
// torch._C._VariableFunctions which is also aliased as Variable._torch
// and also copied into 'torch' module.

#include <Python.h>

#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/Dtype.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/tensor_layouts.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/utils/cuda_lazy_init.h"

#include <ATen/ATen.h>

#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <utility>

using at::Tensor;
using at::Device;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;
using at::IntArrayRef;
using at::Generator;
using at::TensorList;
using at::Dimname;

using namespace torch::autograd::utils;

namespace torch { namespace autograd {

static void check_out_type_matches(Tensor result,
                                   ScalarType scalarType, bool scalarType_is_none,
                                   const THPLayout& layout, bool layout_is_none,
                                   const Device& device, bool device_is_none) {
  if (scalarType_is_none && layout_is_none && device_is_none) {  // common case
    return;
  }
  if (!scalarType_is_none && result.scalar_type() != scalarType) {
    AT_ERROR(
        "dtype ", scalarType,
        " does not match dtype of out parameter (", result.scalar_type(), ")");
  }
  auto scalarType_arg = scalarType_is_none ? result.scalar_type() : scalarType;
  auto layout_arg = layout_is_none ? result.layout() : layout.layout;
  auto device_type_arg = device_is_none ? result.device().type() : device.type();
  if (result.scalar_type() != scalarType_arg) {
    AT_ERROR(
        "scalar type ", scalarType_arg,
        " does not match scalar type of out parameter (", result.scalar_type(), ")");
  }
  if (result.layout() != layout_arg) {
    AT_ERROR(
        "layout ", layout_arg,
        " does not match layout of out parameter (", result.layout(), ")");
  }
  if (result.device().type() != device_type_arg) {
    AT_ERROR(
        "device type ", device_type_arg,
        " does not match device type of out parameter (", result.device().type(), ")");
  }
}

inline Tensor dispatch_arange(Scalar end, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::arange_out(result, end);
}

inline Tensor dispatch_arange(Scalar end, const TensorOptions& options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  return torch::arange(end, options);
}

inline Tensor dispatch_arange(Scalar start, Scalar end, Scalar step, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::arange_out(result, start, end, step);
}

inline Tensor dispatch_arange(Scalar start, Scalar end, Scalar step, const TensorOptions& options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  return torch::arange(start, end, step, options);
}

static PyObject * THPVariable_arange(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arange(Scalar end, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "arange(Scalar start, Scalar end, Scalar step=1, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    if (r.isNone(1)) {
      auto end = r.scalar(0);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      c10::optional<ScalarType> scalarType = r.scalartypeOptional(2);
      const auto options = TensorOptions()
          .dtype(scalarType)
          .device(r.device(4))
          .layout(r.layout(3).layout)
          .requires_grad(r.toBool(6))
          .pinned_memory(r.toBool(5));
      return wrap(dispatch_arange(end, options));
    } else {
      TORCH_CHECK(!r.toBool(5), " `pin_memory` and `out` parameters are incompatible");
      check_out_type_matches(r.tensor(1), r.scalartype(2), r.isNone(2), r.layout(3), r.isNone(3),
                             r.device(4), r.isNone(4));
      return wrap(dispatch_arange(r.scalar(0), r.tensor(1)).set_requires_grad(r.toBool(6)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(3)) {
      auto start = r.scalar(0);
      auto end = r.scalar(1);
      auto step = r.scalar(2);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      c10::optional<ScalarType> scalarType = r.scalartypeOptional(4);
      const auto options = TensorOptions()
          .dtype(scalarType)
          .device(r.device(6))
          .layout(r.layout(5).layout)
          .requires_grad(r.toBool(8))
          .pinned_memory(r.toBool(7));
      return wrap(dispatch_arange(start, end, step, options));
    } else {
      TORCH_CHECK(!r.toBool(7), " `pin_memory` and `out` parameters are incompatible");
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4), r.layout(5), r.isNone(5),
                               r.device(6), r.isNone(6));
      return wrap(dispatch_arange(r.scalar(0), r.scalar(1), r.scalar(2), r.tensor(3)).set_requires_grad(r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

inline Tensor dispatch_range(Scalar start, Scalar end, Scalar step, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(result));
  return at::range_out(result, start, end, step);
}

inline Tensor dispatch_range(Scalar start, Scalar end, Scalar step, const TensorOptions& options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  DeviceGuard device_guard(options.device());
  return torch::range(start, end, step, options);
}

static PyObject * THPVariable_range(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "range(Scalar start, Scalar end, Scalar step=1, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    PyErr_WarnEx(PyExc_UserWarning, "torch.range is deprecated in favor of torch.arange "
        "and will be removed in 0.5. Note that arange generates values in [start; end), "
        "not [start; end].", 1);
    if (r.isNone(3)) {
      const auto options = TensorOptions()
          .dtype(r.scalartype(4))
          .device(r.device(6))
          .layout(r.layout(5).layout)
          .requires_grad(r.toBool(7));
      return wrap(dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), options));
    } else {
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4),
                             r.layout(5), r.isNone(5),
                             r.device(6), r.isNone(6));
      return wrap(dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), r.tensor(3)).set_requires_grad(r.toBool(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

inline Tensor dispatch_randint(int64_t high, IntArrayRef size, Generator * generator, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, high, size, generator);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(high, size, generator, options);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, high, size);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, const TensorOptions & options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(high, size, options);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, Generator * generator, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, low, high, size, generator);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(low, high, size, generator, options);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, low, high, size);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, const TensorOptions & options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(low, high, size, options);
}

static PyObject * THPVariable_randint(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randint(int64_t high, IntArrayRef size, *, Generator generator=None, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "randint(int64_t low, int64_t high, IntArrayRef size, *, Generator generator=None, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  }, /*traceable=*/false);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      auto high = r.toInt64(0);
      auto size = r.intlist(1);
      auto generator = r.generator(2);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      auto dtype = r.scalartypeWithDefault(4, at::ScalarType::Long);
      auto device = r.device(6);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(5).layout)
          .requires_grad(r.toBool(7));
      return wrap(dispatch_randint(high, size, generator, options));
    } else {
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4),
                             r.layout(5), r.isNone(5),
                             r.device(6), r.isNone(6));
      return wrap(dispatch_randint(r.toInt64(0), r.intlist(1), r.generator(2), r.tensor(3)).set_requires_grad(r.toBool(7)));
    }
  } else if (r.idx == 1) {
    if (r.isNone(4)) {
      auto low = r.toInt64(0);
      auto high = r.toInt64(1);
      auto size = r.intlist(2);
      auto generator = r.generator(3);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      auto dtype = r.scalartypeWithDefault(5, at::ScalarType::Long);
      auto device = r.device(7);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(6).layout)
          .requires_grad(r.toBool(8));
      return wrap(dispatch_randint(low, high, size, generator, options));
    } else {
      check_out_type_matches(r.tensor(4), r.scalartype(5), r.isNone(5),
                             r.layout(6), r.isNone(6),
                             r.device(7), r.isNone(7));
      return wrap(dispatch_randint(r.toInt64(0), r.toInt64(1), r.intlist(2), r.generator(3), r.tensor(4)).set_requires_grad(r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// implemented on python object to allow torch.as_tensor to be constructed with arbitrarily nested
// python objects - list, tuple, np array, scalar, etc.
static PyObject * THPVariable_as_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.as_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::as_tensor(torch::tensors::get_default_dispatch_key(), torch::tensors::get_default_scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

// implemented on python object here because PyObject currently not natively declarable
// See: ATen/native/README.md for more context
static PyObject * THPVariable_from_numpy(PyObject* module, PyObject* arg)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.from_numpy", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::tensor_from_numpy(arg));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_nonzero(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero();
}

static Tensor dispatch_nonzero(const Tensor & self, Tensor out) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return at::nonzero_out(out, self);
}

static std::vector<Tensor> dispatch_nonzero_numpy(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero_numpy();
}

static PyObject * THPVariable_nonzero(PyObject* self, PyObject* args, PyObject* kwargs);

static PyObject * THPVariable_sparse_coo_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.sparse_coo_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::sparse_coo_tensor_ctor(torch::tensors::get_default_dispatch_key(), torch::tensors::get_default_scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

// implemented on python object to allow torch.tensor to be constructed with arbitrarily nested
// python objects - list, tuple, np array, scalar, etc.
static PyObject * THPVariable_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::tensor_ctor(torch::tensors::get_default_dispatch_key(), torch::tensors::get_default_scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_get_device(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "get_device(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(r.tensor(0).get_device());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_numel(PyObject* self_, PyObject* args, PyObject* kwargs);

// generated forward declarations start here

static PyObject * THPVariable___and__(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable___lshift__(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable___or__(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable___rshift__(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable___xor__(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__adaptive_avg_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__addr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__addr_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__baddbmm_mkl_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__batch_norm_impl_index(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cast_Byte(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cast_Char(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cast_Double(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cast_Float(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cast_Half(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cast_Int(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cast_Long(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cast_Short(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cat(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__convolution_nogroup(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__copy_from(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cudnn_ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cudnn_init_dropout_state(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cudnn_rnn(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cudnn_rnn_flatten_weight(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cufft_clear_plan_cache(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cufft_get_plan_cache_max_size(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cufft_get_plan_cache_size(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cufft_set_plan_cache_max_size(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__debug_has_internal_overlap(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__dim_arange(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__dirichlet_grad(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__embedding_bag(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__empty_affine_quantized(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__empty_per_channel_affine_quantized(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fft_with_size(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fused_dropout(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__has_compatible_shallow_copy_type(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__index_copy_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__index_put_impl_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__log_softmax_backward_data(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__lu_solve_helper(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__lu_with_info(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__make_per_channel_quantized_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__make_per_tensor_quantized_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__masked_scale(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__max(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__min(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__mkldnn_reshape(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__mkldnn_transpose(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__mkldnn_transpose_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__mode(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__multinomial_alias_draw(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__multinomial_alias_setup(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nnpack_available(PyObject* self_, PyObject* args);
static PyObject * THPVariable__nnpack_spatial_convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__pack_padded_sequence(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__pad_packed_sequence(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__reshape_from_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__s_where(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sample_dirichlet(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__shape_as_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sobol_engine_draw(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sobol_engine_ff_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sobol_engine_initialize_state_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sobol_engine_scramble_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__softmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__softmax_backward_data(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_addmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_mm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_sum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__standard_gamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__standard_gamma_grad(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__std(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__trilinear(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__unique(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__unique2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__use_cudnn_ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__var(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__weight_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__weight_norm_cuda_interface(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_abs(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_abs_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_acos(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_acos_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_adaptive_avg_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_adaptive_max_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_addbmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_addcdiv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_addcmul(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_addmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_addmv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_addmv_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_addr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_affine_grid_generator(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_align_tensors(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_all(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_allclose(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_alpha_dropout(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_alpha_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_angle(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_any(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_argmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_argmin(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_argsort(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_as_strided(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_as_strided_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_asin(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_asin_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_atan(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_atan2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_atan_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_avg_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_baddbmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bartlett_window(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm_backward_elemt(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm_backward_reduce(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm_elemt(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm_gather_stats(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm_gather_stats_with_counts(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm_stats(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm_update_stats(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bernoulli(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bilinear(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_binary_cross_entropy_with_logits(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bincount(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bitwise_and(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bitwise_not(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bitwise_or(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bitwise_xor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_blackman_window(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_broadcast_tensors(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_can_cast(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cartesian_prod(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cat(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cdist(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ceil(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ceil_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_celu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_celu_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_chain_matmul(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cholesky(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cholesky_inverse(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cholesky_solve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_chunk(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clamp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clamp_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clamp_max(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clamp_max_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clamp_min(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clamp_min_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clone(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_combinations(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conj(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_constant_pad_nd(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv_tbc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv_transpose1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv_transpose2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv_transpose3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cos(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cos_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cosh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cosh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cosine_embedding_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cosine_similarity(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cross(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cudnn_affine_grid_generator(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cudnn_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cudnn_convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cudnn_convolution_transpose(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cudnn_grid_sampler(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cudnn_is_acceptable(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cummax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cummin(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cumprod(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cumsum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_dequantize(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_det(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_detach(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_detach_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_diag(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_diag_embed(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_diagflat(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_diagonal(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_digamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_dist(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_div(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_dot(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_dropout(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_eig(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_einsum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_embedding(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_embedding_bag(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_embedding_renorm_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_empty(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_empty_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_empty_strided(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_eq(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_equal(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_erf(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_erf_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_erfc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_erfc_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_erfinv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_exp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_exp_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_expm1(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_expm1_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_eye(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fake_quantize_per_channel_affine(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fake_quantize_per_tensor_affine(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fbgemm_linear_fp16_weight(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fbgemm_linear_fp16_weight_fp32_activation(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fbgemm_linear_int8_weight(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fbgemm_linear_int8_weight_fp32_activation(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fbgemm_linear_quantize_weight(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fbgemm_pack_gemm_matrix_fp16(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fbgemm_pack_quantized_matrix(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_feature_alpha_dropout(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_feature_alpha_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_feature_dropout(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_feature_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fill_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_flatten(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_flip(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_floor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_floor_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_floor_divide(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fmod(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_frac(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_frac_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_frobenius_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_from_file(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_full(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_full_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_gather(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ge(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_geqrf(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ger(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_grid_sampler(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_grid_sampler_2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_grid_sampler_3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_group_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_gru(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_gru_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_gt(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hamming_window(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hann_window(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hardshrink(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hinge_embedding_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_histc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hspmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ifft(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_imag(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_index_add(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_index_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_index_fill(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_index_put(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_index_put_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_index_select(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_instance_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_int_repr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_inverse(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_irfft(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_complex(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_distributed(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_floating_point(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_nonzero(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_same_size(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_signed(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_isclose(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_isfinite(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_isinf(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_isnan(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_kl_div(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_kthvalue(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_layer_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_le(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lerp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lgamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linspace(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log10(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log10_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log1p(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log1p_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log2_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logdet(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logical_and(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logical_not(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logical_or(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logical_xor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logspace(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logsumexp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lstm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lstm_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lstsq(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lt(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lu_solve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_margin_ranking_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_masked_fill(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_masked_scatter(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_masked_select(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_matmul(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_matrix_power(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_matrix_rank(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_max(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_max_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_max_pool1d_with_indices(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_max_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mean(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_median(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_meshgrid(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_min(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_miopen_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_miopen_convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_miopen_convolution_transpose(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_miopen_depthwise_convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_miopen_rnn(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mkldnn_adaptive_avg_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mkldnn_convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mkldnn_convolution_backward_weights(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mkldnn_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mode(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mul(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_multinomial(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mvlgamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_narrow(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_native_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_native_layer_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_native_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ne(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_neg(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_neg_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_norm_except_dim(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_normal(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_nuclear_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ones(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ones_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_orgqr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ormqr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_pairwise_distance(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_pdist(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_pinverse(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_pixel_shuffle(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_poisson(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_poisson_nll_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_polygamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_pow(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_prelu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_prod(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_promote_types(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_q_per_channel_axis(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_q_per_channel_scales(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_q_per_channel_zero_points(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_q_scale(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_q_zero_point(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_qr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantize_per_channel(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantize_per_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_gru(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_gru_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_lstm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_lstm_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_rnn_relu_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_rnn_tanh_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rand(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rand_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_randint_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_randn(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_randn_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_randperm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_real(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_reciprocal(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_reciprocal_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_relu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_relu_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_remainder(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_renorm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_repeat_interleave(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_reshape(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_resize_as_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_result_type(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rfft(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rnn_relu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rnn_relu_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rnn_tanh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rnn_tanh_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_roll(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rot90(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_round(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_round_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rrelu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rrelu_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rsqrt(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rsqrt_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rsub(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_scalar_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_scatter(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_scatter_add(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_select(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_selu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_selu_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sigmoid(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sigmoid_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sign(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sin(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sin_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sinh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sinh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_slogdet(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_smm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_softmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_solve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sort(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_split(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_split_with_sizes(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sqrt(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sqrt_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_square(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_square_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_squeeze(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sspaddmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_stack(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_std(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_std_mean(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_stft(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sub(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_svd(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_symeig(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_t(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_take(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tan(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tan_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tanh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tanh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tensordot(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_threshold(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_threshold_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_topk(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_trace(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_transpose(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_trapz(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_triangular_solve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tril(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tril_indices(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_triplet_margin_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_triu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_triu_indices(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_trunc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_trunc_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unbind(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unique_consecutive(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unique_dim(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unsqueeze(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_var(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_var_mean(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_where(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_zero_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_zeros(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_zeros_like(PyObject* self_, PyObject* args, PyObject* kwargs);

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
static PyMethodDef torch_functions[] = {
  {"arange", (PyCFunction)(void(*)(void))THPVariable_arange, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"as_tensor", (PyCFunction)(void(*)(void))THPVariable_as_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dsmm", (PyCFunction)(void(*)(void))THPVariable_mm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"from_numpy", (PyCFunction)THPVariable_from_numpy, METH_STATIC | METH_O, NULL},
  {"hsmm", (PyCFunction)(void(*)(void))THPVariable_hspmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"nonzero", (PyCFunction)(void(*)(void))THPVariable_nonzero, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randint", (PyCFunction)(void(*)(void))THPVariable_randint, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"range", (PyCFunction)(void(*)(void))THPVariable_range, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"saddmm", (PyCFunction)(void(*)(void))THPVariable_sspaddmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sparse_coo_tensor", (PyCFunction)(void(*)(void))THPVariable_sparse_coo_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"spmm", (PyCFunction)(void(*)(void))THPVariable_mm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tensor", (PyCFunction)(void(*)(void))THPVariable_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"get_device", (PyCFunction)(void(*)(void))THPVariable_get_device, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"numel", (PyCFunction)(void(*)(void))THPVariable_numel, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__and__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___and__>, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__lshift__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___lshift__>, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__or__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___or__>, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__rshift__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___rshift__>, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"__xor__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable___xor__>, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_adaptive_avg_pool2d", (PyCFunction)(void(*)(void))THPVariable__adaptive_avg_pool2d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_addr", (PyCFunction)(void(*)(void))THPVariable__addr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_addr_", (PyCFunction)(void(*)(void))THPVariable__addr_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_baddbmm_mkl_", (PyCFunction)(void(*)(void))THPVariable__baddbmm_mkl_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_batch_norm_impl_index", (PyCFunction)(void(*)(void))THPVariable__batch_norm_impl_index, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Byte", (PyCFunction)(void(*)(void))THPVariable__cast_Byte, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Char", (PyCFunction)(void(*)(void))THPVariable__cast_Char, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Double", (PyCFunction)(void(*)(void))THPVariable__cast_Double, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Float", (PyCFunction)(void(*)(void))THPVariable__cast_Float, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Half", (PyCFunction)(void(*)(void))THPVariable__cast_Half, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Int", (PyCFunction)(void(*)(void))THPVariable__cast_Int, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Long", (PyCFunction)(void(*)(void))THPVariable__cast_Long, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cast_Short", (PyCFunction)(void(*)(void))THPVariable__cast_Short, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cat", (PyCFunction)(void(*)(void))THPVariable__cat, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_convolution", (PyCFunction)(void(*)(void))THPVariable__convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_convolution_nogroup", (PyCFunction)(void(*)(void))THPVariable__convolution_nogroup, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_copy_from", (PyCFunction)(void(*)(void))THPVariable__copy_from, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_ctc_loss", (PyCFunction)(void(*)(void))THPVariable__ctc_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cudnn_ctc_loss", (PyCFunction)(void(*)(void))THPVariable__cudnn_ctc_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cudnn_init_dropout_state", (PyCFunction)(void(*)(void))THPVariable__cudnn_init_dropout_state, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cudnn_rnn", (PyCFunction)(void(*)(void))THPVariable__cudnn_rnn, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cudnn_rnn_flatten_weight", (PyCFunction)(void(*)(void))THPVariable__cudnn_rnn_flatten_weight, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cufft_clear_plan_cache", (PyCFunction)(void(*)(void))THPVariable__cufft_clear_plan_cache, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cufft_get_plan_cache_max_size", (PyCFunction)(void(*)(void))THPVariable__cufft_get_plan_cache_max_size, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cufft_get_plan_cache_size", (PyCFunction)(void(*)(void))THPVariable__cufft_get_plan_cache_size, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_cufft_set_plan_cache_max_size", (PyCFunction)(void(*)(void))THPVariable__cufft_set_plan_cache_max_size, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_debug_has_internal_overlap", (PyCFunction)(void(*)(void))THPVariable__debug_has_internal_overlap, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_dim_arange", (PyCFunction)(void(*)(void))THPVariable__dim_arange, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_dirichlet_grad", (PyCFunction)(void(*)(void))THPVariable__dirichlet_grad, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_embedding_bag", (PyCFunction)(void(*)(void))THPVariable__embedding_bag, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_empty_affine_quantized", (PyCFunction)(void(*)(void))THPVariable__empty_affine_quantized, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_empty_per_channel_affine_quantized", (PyCFunction)(void(*)(void))THPVariable__empty_per_channel_affine_quantized, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_fft_with_size", (PyCFunction)(void(*)(void))THPVariable__fft_with_size, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_fused_dropout", (PyCFunction)(void(*)(void))THPVariable__fused_dropout, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_has_compatible_shallow_copy_type", (PyCFunction)(void(*)(void))THPVariable__has_compatible_shallow_copy_type, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_index_copy_", (PyCFunction)(void(*)(void))THPVariable__index_copy_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_index_put_impl_", (PyCFunction)(void(*)(void))THPVariable__index_put_impl_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_log_softmax", (PyCFunction)(void(*)(void))THPVariable__log_softmax, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_log_softmax_backward_data", (PyCFunction)(void(*)(void))THPVariable__log_softmax_backward_data, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_lu_solve_helper", (PyCFunction)(void(*)(void))THPVariable__lu_solve_helper, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_lu_with_info", (PyCFunction)(void(*)(void))THPVariable__lu_with_info, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_make_per_channel_quantized_tensor", (PyCFunction)(void(*)(void))THPVariable__make_per_channel_quantized_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_make_per_tensor_quantized_tensor", (PyCFunction)(void(*)(void))THPVariable__make_per_tensor_quantized_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_masked_scale", (PyCFunction)(void(*)(void))THPVariable__masked_scale, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_max", (PyCFunction)(void(*)(void))THPVariable__max, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_min", (PyCFunction)(void(*)(void))THPVariable__min, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_mkldnn_reshape", (PyCFunction)(void(*)(void))THPVariable__mkldnn_reshape, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_mkldnn_transpose", (PyCFunction)(void(*)(void))THPVariable__mkldnn_transpose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_mkldnn_transpose_", (PyCFunction)(void(*)(void))THPVariable__mkldnn_transpose_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_mode", (PyCFunction)(void(*)(void))THPVariable__mode, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_multinomial_alias_draw", (PyCFunction)(void(*)(void))THPVariable__multinomial_alias_draw, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_multinomial_alias_setup", (PyCFunction)(void(*)(void))THPVariable__multinomial_alias_setup, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_nnpack_available", (PyCFunction)THPVariable__nnpack_available, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_nnpack_spatial_convolution", (PyCFunction)(void(*)(void))THPVariable__nnpack_spatial_convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_pack_padded_sequence", (PyCFunction)(void(*)(void))THPVariable__pack_padded_sequence, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_pad_packed_sequence", (PyCFunction)(void(*)(void))THPVariable__pad_packed_sequence, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_reshape_from_tensor", (PyCFunction)(void(*)(void))THPVariable__reshape_from_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_s_where", (PyCFunction)(void(*)(void))THPVariable__s_where, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sample_dirichlet", (PyCFunction)(void(*)(void))THPVariable__sample_dirichlet, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_shape_as_tensor", (PyCFunction)(void(*)(void))THPVariable__shape_as_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sobol_engine_draw", (PyCFunction)(void(*)(void))THPVariable__sobol_engine_draw, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sobol_engine_ff_", (PyCFunction)(void(*)(void))THPVariable__sobol_engine_ff_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sobol_engine_initialize_state_", (PyCFunction)(void(*)(void))THPVariable__sobol_engine_initialize_state_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sobol_engine_scramble_", (PyCFunction)(void(*)(void))THPVariable__sobol_engine_scramble_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_softmax", (PyCFunction)(void(*)(void))THPVariable__softmax, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_softmax_backward_data", (PyCFunction)(void(*)(void))THPVariable__softmax_backward_data, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sparse_addmm", (PyCFunction)(void(*)(void))THPVariable__sparse_addmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sparse_mm", (PyCFunction)(void(*)(void))THPVariable__sparse_mm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_sparse_sum", (PyCFunction)(void(*)(void))THPVariable__sparse_sum, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_standard_gamma", (PyCFunction)(void(*)(void))THPVariable__standard_gamma, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_standard_gamma_grad", (PyCFunction)(void(*)(void))THPVariable__standard_gamma_grad, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_std", (PyCFunction)(void(*)(void))THPVariable__std, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_trilinear", (PyCFunction)(void(*)(void))THPVariable__trilinear, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_unique", (PyCFunction)(void(*)(void))THPVariable__unique, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_unique2", (PyCFunction)(void(*)(void))THPVariable__unique2, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_use_cudnn_ctc_loss", (PyCFunction)(void(*)(void))THPVariable__use_cudnn_ctc_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_var", (PyCFunction)(void(*)(void))THPVariable__var, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_weight_norm", (PyCFunction)(void(*)(void))THPVariable__weight_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_weight_norm_cuda_interface", (PyCFunction)(void(*)(void))THPVariable__weight_norm_cuda_interface, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"abs", (PyCFunction)(void(*)(void))THPVariable_abs, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"abs_", (PyCFunction)(void(*)(void))THPVariable_abs_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"acos", (PyCFunction)(void(*)(void))THPVariable_acos, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"acos_", (PyCFunction)(void(*)(void))THPVariable_acos_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"adaptive_avg_pool1d", (PyCFunction)(void(*)(void))THPVariable_adaptive_avg_pool1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"adaptive_max_pool1d", (PyCFunction)(void(*)(void))THPVariable_adaptive_max_pool1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"add", (PyCFunction)(void(*)(void))THPVariable_add, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addbmm", (PyCFunction)(void(*)(void))THPVariable_addbmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addcdiv", (PyCFunction)(void(*)(void))THPVariable_addcdiv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addcmul", (PyCFunction)(void(*)(void))THPVariable_addcmul, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addmm", (PyCFunction)(void(*)(void))THPVariable_addmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addmv", (PyCFunction)(void(*)(void))THPVariable_addmv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addmv_", (PyCFunction)(void(*)(void))THPVariable_addmv_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"addr", (PyCFunction)(void(*)(void))THPVariable_addr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"affine_grid_generator", (PyCFunction)(void(*)(void))THPVariable_affine_grid_generator, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"align_tensors", (PyCFunction)(void(*)(void))THPVariable_align_tensors, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"all", (PyCFunction)(void(*)(void))THPVariable_all, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"allclose", (PyCFunction)(void(*)(void))THPVariable_allclose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"alpha_dropout", (PyCFunction)(void(*)(void))THPVariable_alpha_dropout, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"alpha_dropout_", (PyCFunction)(void(*)(void))THPVariable_alpha_dropout_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"angle", (PyCFunction)(void(*)(void))THPVariable_angle, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"any", (PyCFunction)(void(*)(void))THPVariable_any, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"argmax", (PyCFunction)(void(*)(void))THPVariable_argmax, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"argmin", (PyCFunction)(void(*)(void))THPVariable_argmin, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"argsort", (PyCFunction)(void(*)(void))THPVariable_argsort, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"as_strided", (PyCFunction)(void(*)(void))THPVariable_as_strided, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"as_strided_", (PyCFunction)(void(*)(void))THPVariable_as_strided_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"asin", (PyCFunction)(void(*)(void))THPVariable_asin, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"asin_", (PyCFunction)(void(*)(void))THPVariable_asin_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"atan", (PyCFunction)(void(*)(void))THPVariable_atan, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"atan2", (PyCFunction)(void(*)(void))THPVariable_atan2, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"atan_", (PyCFunction)(void(*)(void))THPVariable_atan_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"avg_pool1d", (PyCFunction)(void(*)(void))THPVariable_avg_pool1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"baddbmm", (PyCFunction)(void(*)(void))THPVariable_baddbmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bartlett_window", (PyCFunction)(void(*)(void))THPVariable_bartlett_window, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"batch_norm", (PyCFunction)(void(*)(void))THPVariable_batch_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"batch_norm_backward_elemt", (PyCFunction)(void(*)(void))THPVariable_batch_norm_backward_elemt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"batch_norm_backward_reduce", (PyCFunction)(void(*)(void))THPVariable_batch_norm_backward_reduce, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"batch_norm_elemt", (PyCFunction)(void(*)(void))THPVariable_batch_norm_elemt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"batch_norm_gather_stats", (PyCFunction)(void(*)(void))THPVariable_batch_norm_gather_stats, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"batch_norm_gather_stats_with_counts", (PyCFunction)(void(*)(void))THPVariable_batch_norm_gather_stats_with_counts, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"batch_norm_stats", (PyCFunction)(void(*)(void))THPVariable_batch_norm_stats, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"batch_norm_update_stats", (PyCFunction)(void(*)(void))THPVariable_batch_norm_update_stats, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bernoulli", (PyCFunction)(void(*)(void))THPVariable_bernoulli, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bilinear", (PyCFunction)(void(*)(void))THPVariable_bilinear, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"binary_cross_entropy_with_logits", (PyCFunction)(void(*)(void))THPVariable_binary_cross_entropy_with_logits, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bincount", (PyCFunction)(void(*)(void))THPVariable_bincount, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bitwise_and", (PyCFunction)(void(*)(void))THPVariable_bitwise_and, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bitwise_not", (PyCFunction)(void(*)(void))THPVariable_bitwise_not, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bitwise_or", (PyCFunction)(void(*)(void))THPVariable_bitwise_or, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bitwise_xor", (PyCFunction)(void(*)(void))THPVariable_bitwise_xor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"blackman_window", (PyCFunction)(void(*)(void))THPVariable_blackman_window, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"bmm", (PyCFunction)(void(*)(void))THPVariable_bmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"broadcast_tensors", (PyCFunction)(void(*)(void))THPVariable_broadcast_tensors, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"can_cast", (PyCFunction)(void(*)(void))THPVariable_can_cast, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cartesian_prod", (PyCFunction)(void(*)(void))THPVariable_cartesian_prod, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cat", (PyCFunction)(void(*)(void))THPVariable_cat, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cdist", (PyCFunction)(void(*)(void))THPVariable_cdist, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ceil", (PyCFunction)(void(*)(void))THPVariable_ceil, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ceil_", (PyCFunction)(void(*)(void))THPVariable_ceil_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"celu", (PyCFunction)(void(*)(void))THPVariable_celu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"celu_", (PyCFunction)(void(*)(void))THPVariable_celu_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"chain_matmul", (PyCFunction)(void(*)(void))THPVariable_chain_matmul, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cholesky", (PyCFunction)(void(*)(void))THPVariable_cholesky, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cholesky_inverse", (PyCFunction)(void(*)(void))THPVariable_cholesky_inverse, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cholesky_solve", (PyCFunction)(void(*)(void))THPVariable_cholesky_solve, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"chunk", (PyCFunction)(void(*)(void))THPVariable_chunk, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"clamp", (PyCFunction)(void(*)(void))THPVariable_clamp, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"clamp_", (PyCFunction)(void(*)(void))THPVariable_clamp_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"clamp_max", (PyCFunction)(void(*)(void))THPVariable_clamp_max, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"clamp_max_", (PyCFunction)(void(*)(void))THPVariable_clamp_max_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"clamp_min", (PyCFunction)(void(*)(void))THPVariable_clamp_min, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"clamp_min_", (PyCFunction)(void(*)(void))THPVariable_clamp_min_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"clone", (PyCFunction)(void(*)(void))THPVariable_clone, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"combinations", (PyCFunction)(void(*)(void))THPVariable_combinations, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conj", (PyCFunction)(void(*)(void))THPVariable_conj, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"constant_pad_nd", (PyCFunction)(void(*)(void))THPVariable_constant_pad_nd, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv1d", (PyCFunction)(void(*)(void))THPVariable_conv1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv2d", (PyCFunction)(void(*)(void))THPVariable_conv2d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv3d", (PyCFunction)(void(*)(void))THPVariable_conv3d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv_tbc", (PyCFunction)(void(*)(void))THPVariable_conv_tbc, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv_transpose1d", (PyCFunction)(void(*)(void))THPVariable_conv_transpose1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv_transpose2d", (PyCFunction)(void(*)(void))THPVariable_conv_transpose2d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"conv_transpose3d", (PyCFunction)(void(*)(void))THPVariable_conv_transpose3d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"convolution", (PyCFunction)(void(*)(void))THPVariable_convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cos", (PyCFunction)(void(*)(void))THPVariable_cos, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cos_", (PyCFunction)(void(*)(void))THPVariable_cos_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cosh", (PyCFunction)(void(*)(void))THPVariable_cosh, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cosh_", (PyCFunction)(void(*)(void))THPVariable_cosh_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cosine_embedding_loss", (PyCFunction)(void(*)(void))THPVariable_cosine_embedding_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cosine_similarity", (PyCFunction)(void(*)(void))THPVariable_cosine_similarity, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cross", (PyCFunction)(void(*)(void))THPVariable_cross, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ctc_loss", (PyCFunction)(void(*)(void))THPVariable_ctc_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_affine_grid_generator", (PyCFunction)(void(*)(void))THPVariable_cudnn_affine_grid_generator, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_batch_norm", (PyCFunction)(void(*)(void))THPVariable_cudnn_batch_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_convolution", (PyCFunction)(void(*)(void))THPVariable_cudnn_convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_convolution_transpose", (PyCFunction)(void(*)(void))THPVariable_cudnn_convolution_transpose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_grid_sampler", (PyCFunction)(void(*)(void))THPVariable_cudnn_grid_sampler, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cudnn_is_acceptable", (PyCFunction)(void(*)(void))THPVariable_cudnn_is_acceptable, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cummax", (PyCFunction)(void(*)(void))THPVariable_cummax, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cummin", (PyCFunction)(void(*)(void))THPVariable_cummin, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cumprod", (PyCFunction)(void(*)(void))THPVariable_cumprod, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"cumsum", (PyCFunction)(void(*)(void))THPVariable_cumsum, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dequantize", (PyCFunction)(void(*)(void))THPVariable_dequantize, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"det", (PyCFunction)(void(*)(void))THPVariable_det, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"detach", (PyCFunction)(void(*)(void))THPVariable_detach, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"detach_", (PyCFunction)(void(*)(void))THPVariable_detach_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"diag", (PyCFunction)(void(*)(void))THPVariable_diag, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"diag_embed", (PyCFunction)(void(*)(void))THPVariable_diag_embed, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"diagflat", (PyCFunction)(void(*)(void))THPVariable_diagflat, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"diagonal", (PyCFunction)(void(*)(void))THPVariable_diagonal, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"digamma", (PyCFunction)(void(*)(void))THPVariable_digamma, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dist", (PyCFunction)(void(*)(void))THPVariable_dist, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"div", (PyCFunction)(void(*)(void))THPVariable_div, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dot", (PyCFunction)(void(*)(void))THPVariable_dot, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dropout", (PyCFunction)(void(*)(void))THPVariable_dropout, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"dropout_", (PyCFunction)(void(*)(void))THPVariable_dropout_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"eig", (PyCFunction)(void(*)(void))THPVariable_eig, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"einsum", (PyCFunction)(void(*)(void))THPVariable_einsum, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"embedding", (PyCFunction)(void(*)(void))THPVariable_embedding, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"embedding_bag", (PyCFunction)(void(*)(void))THPVariable_embedding_bag, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"embedding_renorm_", (PyCFunction)(void(*)(void))THPVariable_embedding_renorm_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"empty", (PyCFunction)(void(*)(void))THPVariable_empty, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"empty_like", (PyCFunction)(void(*)(void))THPVariable_empty_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"empty_strided", (PyCFunction)(void(*)(void))THPVariable_empty_strided, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"eq", (PyCFunction)(void(*)(void))THPVariable_eq, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"equal", (PyCFunction)(void(*)(void))THPVariable_equal, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"erf", (PyCFunction)(void(*)(void))THPVariable_erf, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"erf_", (PyCFunction)(void(*)(void))THPVariable_erf_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"erfc", (PyCFunction)(void(*)(void))THPVariable_erfc, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"erfc_", (PyCFunction)(void(*)(void))THPVariable_erfc_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"erfinv", (PyCFunction)(void(*)(void))THPVariable_erfinv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"exp", (PyCFunction)(void(*)(void))THPVariable_exp, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"exp_", (PyCFunction)(void(*)(void))THPVariable_exp_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"expm1", (PyCFunction)(void(*)(void))THPVariable_expm1, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"expm1_", (PyCFunction)(void(*)(void))THPVariable_expm1_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"eye", (PyCFunction)(void(*)(void))THPVariable_eye, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fake_quantize_per_channel_affine", (PyCFunction)(void(*)(void))THPVariable_fake_quantize_per_channel_affine, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fake_quantize_per_tensor_affine", (PyCFunction)(void(*)(void))THPVariable_fake_quantize_per_tensor_affine, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fbgemm_linear_fp16_weight", (PyCFunction)(void(*)(void))THPVariable_fbgemm_linear_fp16_weight, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fbgemm_linear_fp16_weight_fp32_activation", (PyCFunction)(void(*)(void))THPVariable_fbgemm_linear_fp16_weight_fp32_activation, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fbgemm_linear_int8_weight", (PyCFunction)(void(*)(void))THPVariable_fbgemm_linear_int8_weight, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fbgemm_linear_int8_weight_fp32_activation", (PyCFunction)(void(*)(void))THPVariable_fbgemm_linear_int8_weight_fp32_activation, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fbgemm_linear_quantize_weight", (PyCFunction)(void(*)(void))THPVariable_fbgemm_linear_quantize_weight, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fbgemm_pack_gemm_matrix_fp16", (PyCFunction)(void(*)(void))THPVariable_fbgemm_pack_gemm_matrix_fp16, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fbgemm_pack_quantized_matrix", (PyCFunction)(void(*)(void))THPVariable_fbgemm_pack_quantized_matrix, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"feature_alpha_dropout", (PyCFunction)(void(*)(void))THPVariable_feature_alpha_dropout, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"feature_alpha_dropout_", (PyCFunction)(void(*)(void))THPVariable_feature_alpha_dropout_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"feature_dropout", (PyCFunction)(void(*)(void))THPVariable_feature_dropout, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"feature_dropout_", (PyCFunction)(void(*)(void))THPVariable_feature_dropout_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fft", (PyCFunction)(void(*)(void))THPVariable_fft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fill_", (PyCFunction)(void(*)(void))THPVariable_fill_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"flatten", (PyCFunction)(void(*)(void))THPVariable_flatten, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"flip", (PyCFunction)(void(*)(void))THPVariable_flip, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"floor", (PyCFunction)(void(*)(void))THPVariable_floor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"floor_", (PyCFunction)(void(*)(void))THPVariable_floor_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"floor_divide", (PyCFunction)(void(*)(void))THPVariable_floor_divide, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"fmod", (PyCFunction)(void(*)(void))THPVariable_fmod, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"frac", (PyCFunction)(void(*)(void))THPVariable_frac, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"frac_", (PyCFunction)(void(*)(void))THPVariable_frac_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"frobenius_norm", (PyCFunction)(void(*)(void))THPVariable_frobenius_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"from_file", (PyCFunction)(void(*)(void))THPVariable_from_file, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"full", (PyCFunction)(void(*)(void))THPVariable_full, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"full_like", (PyCFunction)(void(*)(void))THPVariable_full_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"gather", (PyCFunction)(void(*)(void))THPVariable_gather, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ge", (PyCFunction)(void(*)(void))THPVariable_ge, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"geqrf", (PyCFunction)(void(*)(void))THPVariable_geqrf, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ger", (PyCFunction)(void(*)(void))THPVariable_ger, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"grid_sampler", (PyCFunction)(void(*)(void))THPVariable_grid_sampler, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"grid_sampler_2d", (PyCFunction)(void(*)(void))THPVariable_grid_sampler_2d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"grid_sampler_3d", (PyCFunction)(void(*)(void))THPVariable_grid_sampler_3d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"group_norm", (PyCFunction)(void(*)(void))THPVariable_group_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"gru", (PyCFunction)(void(*)(void))THPVariable_gru, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"gru_cell", (PyCFunction)(void(*)(void))THPVariable_gru_cell, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"gt", (PyCFunction)(void(*)(void))THPVariable_gt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"hamming_window", (PyCFunction)(void(*)(void))THPVariable_hamming_window, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"hann_window", (PyCFunction)(void(*)(void))THPVariable_hann_window, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"hardshrink", (PyCFunction)(void(*)(void))THPVariable_hardshrink, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"hinge_embedding_loss", (PyCFunction)(void(*)(void))THPVariable_hinge_embedding_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"histc", (PyCFunction)(void(*)(void))THPVariable_histc, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"hspmm", (PyCFunction)(void(*)(void))THPVariable_hspmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ifft", (PyCFunction)(void(*)(void))THPVariable_ifft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"imag", (PyCFunction)(void(*)(void))THPVariable_imag, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"index_add", (PyCFunction)(void(*)(void))THPVariable_index_add, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"index_copy", (PyCFunction)(void(*)(void))THPVariable_index_copy, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"index_fill", (PyCFunction)(void(*)(void))THPVariable_index_fill, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"index_put", (PyCFunction)(void(*)(void))THPVariable_index_put, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"index_put_", (PyCFunction)(void(*)(void))THPVariable_index_put_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"index_select", (PyCFunction)(void(*)(void))THPVariable_index_select, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"instance_norm", (PyCFunction)(void(*)(void))THPVariable_instance_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"int_repr", (PyCFunction)(void(*)(void))THPVariable_int_repr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"inverse", (PyCFunction)(void(*)(void))THPVariable_inverse, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"irfft", (PyCFunction)(void(*)(void))THPVariable_irfft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_complex", (PyCFunction)(void(*)(void))THPVariable_is_complex, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_distributed", (PyCFunction)(void(*)(void))THPVariable_is_distributed, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_floating_point", (PyCFunction)(void(*)(void))THPVariable_is_floating_point, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_nonzero", (PyCFunction)(void(*)(void))THPVariable_is_nonzero, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_same_size", (PyCFunction)(void(*)(void))THPVariable_is_same_size, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"is_signed", (PyCFunction)(void(*)(void))THPVariable_is_signed, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"isclose", (PyCFunction)(void(*)(void))THPVariable_isclose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"isfinite", (PyCFunction)(void(*)(void))THPVariable_isfinite, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"isinf", (PyCFunction)(void(*)(void))THPVariable_isinf, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"isnan", (PyCFunction)(void(*)(void))THPVariable_isnan, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"kl_div", (PyCFunction)(void(*)(void))THPVariable_kl_div, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"kthvalue", (PyCFunction)(void(*)(void))THPVariable_kthvalue, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"layer_norm", (PyCFunction)(void(*)(void))THPVariable_layer_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"le", (PyCFunction)(void(*)(void))THPVariable_le, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lerp", (PyCFunction)(void(*)(void))THPVariable_lerp, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lgamma", (PyCFunction)(void(*)(void))THPVariable_lgamma, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"linspace", (PyCFunction)(void(*)(void))THPVariable_linspace, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log", (PyCFunction)(void(*)(void))THPVariable_log, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log10", (PyCFunction)(void(*)(void))THPVariable_log10, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log10_", (PyCFunction)(void(*)(void))THPVariable_log10_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log1p", (PyCFunction)(void(*)(void))THPVariable_log1p, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log1p_", (PyCFunction)(void(*)(void))THPVariable_log1p_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log2", (PyCFunction)(void(*)(void))THPVariable_log2, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log2_", (PyCFunction)(void(*)(void))THPVariable_log2_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log_", (PyCFunction)(void(*)(void))THPVariable_log_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"log_softmax", (PyCFunction)(void(*)(void))THPVariable_log_softmax, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"logdet", (PyCFunction)(void(*)(void))THPVariable_logdet, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"logical_and", (PyCFunction)(void(*)(void))THPVariable_logical_and, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"logical_not", (PyCFunction)(void(*)(void))THPVariable_logical_not, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"logical_or", (PyCFunction)(void(*)(void))THPVariable_logical_or, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"logical_xor", (PyCFunction)(void(*)(void))THPVariable_logical_xor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"logspace", (PyCFunction)(void(*)(void))THPVariable_logspace, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"logsumexp", (PyCFunction)(void(*)(void))THPVariable_logsumexp, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lstm", (PyCFunction)(void(*)(void))THPVariable_lstm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lstm_cell", (PyCFunction)(void(*)(void))THPVariable_lstm_cell, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lstsq", (PyCFunction)(void(*)(void))THPVariable_lstsq, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lt", (PyCFunction)(void(*)(void))THPVariable_lt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"lu_solve", (PyCFunction)(void(*)(void))THPVariable_lu_solve, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"margin_ranking_loss", (PyCFunction)(void(*)(void))THPVariable_margin_ranking_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"masked_fill", (PyCFunction)(void(*)(void))THPVariable_masked_fill, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"masked_scatter", (PyCFunction)(void(*)(void))THPVariable_masked_scatter, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"masked_select", (PyCFunction)(void(*)(void))THPVariable_masked_select, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"matmul", (PyCFunction)(void(*)(void))THPVariable_matmul, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"matrix_power", (PyCFunction)(void(*)(void))THPVariable_matrix_power, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"matrix_rank", (PyCFunction)(void(*)(void))THPVariable_matrix_rank, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"max", (PyCFunction)(void(*)(void))THPVariable_max, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"max_pool1d", (PyCFunction)(void(*)(void))THPVariable_max_pool1d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"max_pool1d_with_indices", (PyCFunction)(void(*)(void))THPVariable_max_pool1d_with_indices, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"max_pool2d", (PyCFunction)(void(*)(void))THPVariable_max_pool2d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"max_pool3d", (PyCFunction)(void(*)(void))THPVariable_max_pool3d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mean", (PyCFunction)(void(*)(void))THPVariable_mean, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"median", (PyCFunction)(void(*)(void))THPVariable_median, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"meshgrid", (PyCFunction)(void(*)(void))THPVariable_meshgrid, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"min", (PyCFunction)(void(*)(void))THPVariable_min, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"miopen_batch_norm", (PyCFunction)(void(*)(void))THPVariable_miopen_batch_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"miopen_convolution", (PyCFunction)(void(*)(void))THPVariable_miopen_convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"miopen_convolution_transpose", (PyCFunction)(void(*)(void))THPVariable_miopen_convolution_transpose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"miopen_depthwise_convolution", (PyCFunction)(void(*)(void))THPVariable_miopen_depthwise_convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"miopen_rnn", (PyCFunction)(void(*)(void))THPVariable_miopen_rnn, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mkldnn_adaptive_avg_pool2d", (PyCFunction)(void(*)(void))THPVariable_mkldnn_adaptive_avg_pool2d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mkldnn_convolution", (PyCFunction)(void(*)(void))THPVariable_mkldnn_convolution, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mkldnn_convolution_backward_weights", (PyCFunction)(void(*)(void))THPVariable_mkldnn_convolution_backward_weights, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mkldnn_max_pool2d", (PyCFunction)(void(*)(void))THPVariable_mkldnn_max_pool2d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mm", (PyCFunction)(void(*)(void))THPVariable_mm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mode", (PyCFunction)(void(*)(void))THPVariable_mode, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mul", (PyCFunction)(void(*)(void))THPVariable_mul, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"multinomial", (PyCFunction)(void(*)(void))THPVariable_multinomial, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mv", (PyCFunction)(void(*)(void))THPVariable_mv, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"mvlgamma", (PyCFunction)(void(*)(void))THPVariable_mvlgamma, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"narrow", (PyCFunction)(void(*)(void))THPVariable_narrow, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"native_batch_norm", (PyCFunction)(void(*)(void))THPVariable_native_batch_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"native_layer_norm", (PyCFunction)(void(*)(void))THPVariable_native_layer_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"native_norm", (PyCFunction)(void(*)(void))THPVariable_native_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ne", (PyCFunction)(void(*)(void))THPVariable_ne, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"neg", (PyCFunction)(void(*)(void))THPVariable_neg, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"neg_", (PyCFunction)(void(*)(void))THPVariable_neg_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"norm", (PyCFunction)(void(*)(void))THPVariable_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"norm_except_dim", (PyCFunction)(void(*)(void))THPVariable_norm_except_dim, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"normal", (PyCFunction)(void(*)(void))THPVariable_normal, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"nuclear_norm", (PyCFunction)(void(*)(void))THPVariable_nuclear_norm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ones", (PyCFunction)(void(*)(void))THPVariable_ones, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ones_like", (PyCFunction)(void(*)(void))THPVariable_ones_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"orgqr", (PyCFunction)(void(*)(void))THPVariable_orgqr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"ormqr", (PyCFunction)(void(*)(void))THPVariable_ormqr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pairwise_distance", (PyCFunction)(void(*)(void))THPVariable_pairwise_distance, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pdist", (PyCFunction)(void(*)(void))THPVariable_pdist, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pinverse", (PyCFunction)(void(*)(void))THPVariable_pinverse, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pixel_shuffle", (PyCFunction)(void(*)(void))THPVariable_pixel_shuffle, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"poisson", (PyCFunction)(void(*)(void))THPVariable_poisson, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"poisson_nll_loss", (PyCFunction)(void(*)(void))THPVariable_poisson_nll_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"polygamma", (PyCFunction)(void(*)(void))THPVariable_polygamma, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"pow", (PyCFunction)(void(*)(void))THPVariable_pow, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"prelu", (PyCFunction)(void(*)(void))THPVariable_prelu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"prod", (PyCFunction)(void(*)(void))THPVariable_prod, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"promote_types", (PyCFunction)(void(*)(void))THPVariable_promote_types, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"q_per_channel_axis", (PyCFunction)(void(*)(void))THPVariable_q_per_channel_axis, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"q_per_channel_scales", (PyCFunction)(void(*)(void))THPVariable_q_per_channel_scales, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"q_per_channel_zero_points", (PyCFunction)(void(*)(void))THPVariable_q_per_channel_zero_points, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"q_scale", (PyCFunction)(void(*)(void))THPVariable_q_scale, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"q_zero_point", (PyCFunction)(void(*)(void))THPVariable_q_zero_point, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"qr", (PyCFunction)(void(*)(void))THPVariable_qr, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"quantize_per_channel", (PyCFunction)(void(*)(void))THPVariable_quantize_per_channel, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"quantize_per_tensor", (PyCFunction)(void(*)(void))THPVariable_quantize_per_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"quantized_gru", (PyCFunction)(void(*)(void))THPVariable_quantized_gru, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"quantized_gru_cell", (PyCFunction)(void(*)(void))THPVariable_quantized_gru_cell, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"quantized_lstm", (PyCFunction)(void(*)(void))THPVariable_quantized_lstm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"quantized_lstm_cell", (PyCFunction)(void(*)(void))THPVariable_quantized_lstm_cell, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"quantized_max_pool2d", (PyCFunction)(void(*)(void))THPVariable_quantized_max_pool2d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"quantized_rnn_relu_cell", (PyCFunction)(void(*)(void))THPVariable_quantized_rnn_relu_cell, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"quantized_rnn_tanh_cell", (PyCFunction)(void(*)(void))THPVariable_quantized_rnn_tanh_cell, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rand", (PyCFunction)(void(*)(void))THPVariable_rand, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rand_like", (PyCFunction)(void(*)(void))THPVariable_rand_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randint_like", (PyCFunction)(void(*)(void))THPVariable_randint_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randn", (PyCFunction)(void(*)(void))THPVariable_randn, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randn_like", (PyCFunction)(void(*)(void))THPVariable_randn_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randperm", (PyCFunction)(void(*)(void))THPVariable_randperm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"real", (PyCFunction)(void(*)(void))THPVariable_real, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"reciprocal", (PyCFunction)(void(*)(void))THPVariable_reciprocal, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"reciprocal_", (PyCFunction)(void(*)(void))THPVariable_reciprocal_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"relu", (PyCFunction)(void(*)(void))THPVariable_relu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"relu_", (PyCFunction)(void(*)(void))THPVariable_relu_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"remainder", (PyCFunction)(void(*)(void))THPVariable_remainder, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"renorm", (PyCFunction)(void(*)(void))THPVariable_renorm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"repeat_interleave", (PyCFunction)(void(*)(void))THPVariable_repeat_interleave, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"reshape", (PyCFunction)(void(*)(void))THPVariable_reshape, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"resize_as_", (PyCFunction)(void(*)(void))THPVariable_resize_as_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"result_type", (PyCFunction)(void(*)(void))THPVariable_result_type, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rfft", (PyCFunction)(void(*)(void))THPVariable_rfft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rnn_relu", (PyCFunction)(void(*)(void))THPVariable_rnn_relu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rnn_relu_cell", (PyCFunction)(void(*)(void))THPVariable_rnn_relu_cell, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rnn_tanh", (PyCFunction)(void(*)(void))THPVariable_rnn_tanh, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rnn_tanh_cell", (PyCFunction)(void(*)(void))THPVariable_rnn_tanh_cell, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"roll", (PyCFunction)(void(*)(void))THPVariable_roll, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rot90", (PyCFunction)(void(*)(void))THPVariable_rot90, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"round", (PyCFunction)(void(*)(void))THPVariable_round, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"round_", (PyCFunction)(void(*)(void))THPVariable_round_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rrelu", (PyCFunction)(void(*)(void))THPVariable_rrelu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rrelu_", (PyCFunction)(void(*)(void))THPVariable_rrelu_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rsqrt", (PyCFunction)(void(*)(void))THPVariable_rsqrt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rsqrt_", (PyCFunction)(void(*)(void))THPVariable_rsqrt_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rsub", (PyCFunction)(void(*)(void))THPVariable_rsub, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"scalar_tensor", (PyCFunction)(void(*)(void))THPVariable_scalar_tensor, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"scatter", (PyCFunction)(void(*)(void))THPVariable_scatter, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"scatter_add", (PyCFunction)(void(*)(void))THPVariable_scatter_add, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"select", (PyCFunction)(void(*)(void))THPVariable_select, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"selu", (PyCFunction)(void(*)(void))THPVariable_selu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"selu_", (PyCFunction)(void(*)(void))THPVariable_selu_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sigmoid", (PyCFunction)(void(*)(void))THPVariable_sigmoid, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sigmoid_", (PyCFunction)(void(*)(void))THPVariable_sigmoid_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sign", (PyCFunction)(void(*)(void))THPVariable_sign, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sin", (PyCFunction)(void(*)(void))THPVariable_sin, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sin_", (PyCFunction)(void(*)(void))THPVariable_sin_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sinh", (PyCFunction)(void(*)(void))THPVariable_sinh, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sinh_", (PyCFunction)(void(*)(void))THPVariable_sinh_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"slogdet", (PyCFunction)(void(*)(void))THPVariable_slogdet, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"smm", (PyCFunction)(void(*)(void))THPVariable_smm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"softmax", (PyCFunction)(void(*)(void))THPVariable_softmax, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"solve", (PyCFunction)(void(*)(void))THPVariable_solve, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sort", (PyCFunction)(void(*)(void))THPVariable_sort, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"split", (PyCFunction)(void(*)(void))THPVariable_split, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"split_with_sizes", (PyCFunction)(void(*)(void))THPVariable_split_with_sizes, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sqrt", (PyCFunction)(void(*)(void))THPVariable_sqrt, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sqrt_", (PyCFunction)(void(*)(void))THPVariable_sqrt_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"square", (PyCFunction)(void(*)(void))THPVariable_square, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"square_", (PyCFunction)(void(*)(void))THPVariable_square_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"squeeze", (PyCFunction)(void(*)(void))THPVariable_squeeze, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sspaddmm", (PyCFunction)(void(*)(void))THPVariable_sspaddmm, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"stack", (PyCFunction)(void(*)(void))THPVariable_stack, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"std", (PyCFunction)(void(*)(void))THPVariable_std, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"std_mean", (PyCFunction)(void(*)(void))THPVariable_std_mean, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"stft", (PyCFunction)(void(*)(void))THPVariable_stft, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sub", (PyCFunction)(void(*)(void))THPVariable_sub, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"sum", (PyCFunction)(void(*)(void))THPVariable_sum, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"svd", (PyCFunction)(void(*)(void))THPVariable_svd, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"symeig", (PyCFunction)(void(*)(void))THPVariable_symeig, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"t", (PyCFunction)(void(*)(void))THPVariable_t, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"take", (PyCFunction)(void(*)(void))THPVariable_take, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tan", (PyCFunction)(void(*)(void))THPVariable_tan, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tan_", (PyCFunction)(void(*)(void))THPVariable_tan_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tanh", (PyCFunction)(void(*)(void))THPVariable_tanh, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tanh_", (PyCFunction)(void(*)(void))THPVariable_tanh_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tensordot", (PyCFunction)(void(*)(void))THPVariable_tensordot, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"threshold", (PyCFunction)(void(*)(void))THPVariable_threshold, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"threshold_", (PyCFunction)(void(*)(void))THPVariable_threshold_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"topk", (PyCFunction)(void(*)(void))THPVariable_topk, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"trace", (PyCFunction)(void(*)(void))THPVariable_trace, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"transpose", (PyCFunction)(void(*)(void))THPVariable_transpose, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"trapz", (PyCFunction)(void(*)(void))THPVariable_trapz, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"triangular_solve", (PyCFunction)(void(*)(void))THPVariable_triangular_solve, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tril", (PyCFunction)(void(*)(void))THPVariable_tril, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"tril_indices", (PyCFunction)(void(*)(void))THPVariable_tril_indices, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"triplet_margin_loss", (PyCFunction)(void(*)(void))THPVariable_triplet_margin_loss, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"triu", (PyCFunction)(void(*)(void))THPVariable_triu, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"triu_indices", (PyCFunction)(void(*)(void))THPVariable_triu_indices, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"trunc", (PyCFunction)(void(*)(void))THPVariable_trunc, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"trunc_", (PyCFunction)(void(*)(void))THPVariable_trunc_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"unbind", (PyCFunction)(void(*)(void))THPVariable_unbind, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"unique_consecutive", (PyCFunction)(void(*)(void))THPVariable_unique_consecutive, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"unique_dim", (PyCFunction)(void(*)(void))THPVariable_unique_dim, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"unsqueeze", (PyCFunction)(void(*)(void))THPVariable_unsqueeze, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"var", (PyCFunction)(void(*)(void))THPVariable_var, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"var_mean", (PyCFunction)(void(*)(void))THPVariable_var_mean, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"where", (PyCFunction)(void(*)(void))THPVariable_where, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"zero_", (PyCFunction)(void(*)(void))THPVariable_zero_, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"zeros", (PyCFunction)(void(*)(void))THPVariable_zeros, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"zeros_like", (PyCFunction)(void(*)(void))THPVariable_zeros_like, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {NULL}
};

static PyTypeObject THPVariableFunctions = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._VariableFunctions",         /* tp_name */
  0,                                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_vectorcall_offset */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  torch_functions,                       /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  0                                      /* tp_new */
};

void initTorchFunctions(PyObject* module) {
  if (PyType_Ready(&THPVariableFunctions) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPVariableFunctions);
  if (PyModule_AddObject(module, "_VariableFunctions", (PyObject*)&THPVariableFunctions) < 0) {
    throw python_error();
  }
}

/*
 *
 * Calls __torch_function__ on the overloaded arguments to a torch API
 * function in order of precedence, returning the first result that is
 * not NotImplemented. If all arguments return NotImplemented, raises a
 * TypeError.
 *
 * Assumes overloaded_args has at least one entry. All entries must have
 * a __torch_function__ attribute that resolves to a callable that
 * accepts a torch API function, arguments, and keyword arguments for
 * the torch API function.
 *
 * It is sufficient to call PythonArgs::has_torch_function before
 * calling this function to verify that there are valid arguments
 * present. If that is not done then special care must be taken to
 * ensure there are arguments that are overloaded with
 * __torch_function__.
 *
 * See torch._overrides._implement_torch_function for the equivalent
 * code in the pure-python implementation.
 *
 * 'r' is a parsed PythonArgs instance, returned from
 * PythonArgParser::parse.
 *
 * 'args' is a reference to the python tuple of arguments to the torch
 * API function.
 *
 * 'kwargs' is a reference to the python dict of keyword arguments to
 * the torch API function.
 *
 * 'torch_api' is a reference to python torch API namespace.
 *
 */

PyObject* handle_torch_function(PythonArgs &r, PyObject* args, PyObject* kwargs, PyTypeObject &torch_api) {
  py::object torch_api_function = PyObject_FastGetAttrString((PyObject*)&torch_api, const_cast<char*>(r.get_func_name().data()));
  TORCH_INTERNAL_ASSERT(torch_api_function.ptr() != NULL, "torch API function must exist");
  py::object ret;
  for (auto &arg : r.signature.overloaded_args) {
    py::object torch_function = PyObject_FastGetAttrString(arg.ptr(), "__torch_function__");
    ret = py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(torch_function.ptr(), torch_api_function.ptr(), args, kwargs, NULL));
    if (ret.ptr() != Py_NotImplemented) {
      // Return the reference to the result. This also covers the case where ret
      // is NULL and __torch_function__ raised an exception, which we throw below
      break;
    }
  }
  if (ret.ptr() == nullptr) {
    // if an exception occurred in a user's implementation of
    // __array_function__, throw it
    throw python_error();
  }
  else if (ret.ptr() == Py_NotImplemented) {
    // all __torch_function__ implementations in overloaded_args
    // returned NotImplemented, so we raise a TypeError.
    std::stringstream ss;
    ss << "no implementation found for 'torch." << r.get_func_name()
       << "' on types that implement __torch_function__: [";
    for (auto &arg : r.signature.overloaded_args) {
      ss << arg.ptr()->ob_type->tp_name;
      if (!arg.is(r.signature.overloaded_args.back())) {
        ss << ", ";
      }
      else {
        ss << "]";
      }
    }
    const std::string& tmp = ss.str();
    PyErr_SetString(PyExc_TypeError, tmp.c_str());
    throw python_error();
  }
  return ret.release().ptr();
}

// generated methods start here

\
// __and__
static PyObject * THPVariable___and__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__and__(Tensor input, Tensor other)",
    "__and__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch___and__ = [](const Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__and__(other);
      };
      return wrap(dispatch___and__(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch___and__ = [](const Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__and__(other);
      };
      return wrap(dispatch___and__(_r.tensor(0), _r.scalar(1)));
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
  static PythonArgParser parser({
    "__lshift__(Tensor input, Tensor other)",
    "__lshift__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch___lshift__ = [](const Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__lshift__(other);
      };
      return wrap(dispatch___lshift__(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch___lshift__ = [](const Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__lshift__(other);
      };
      return wrap(dispatch___lshift__(_r.tensor(0), _r.scalar(1)));
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
  static PythonArgParser parser({
    "__or__(Tensor input, Tensor other)",
    "__or__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch___or__ = [](const Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__or__(other);
      };
      return wrap(dispatch___or__(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch___or__ = [](const Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__or__(other);
      };
      return wrap(dispatch___or__(_r.tensor(0), _r.scalar(1)));
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
  static PythonArgParser parser({
    "__rshift__(Tensor input, Tensor other)",
    "__rshift__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch___rshift__ = [](const Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__rshift__(other);
      };
      return wrap(dispatch___rshift__(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch___rshift__ = [](const Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__rshift__(other);
      };
      return wrap(dispatch___rshift__(_r.tensor(0), _r.scalar(1)));
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
  static PythonArgParser parser({
    "__xor__(Tensor input, Tensor other)",
    "__xor__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor
      auto dispatch___xor__ = [](const Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__xor__(other);
      };
      return wrap(dispatch___xor__(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor
      auto dispatch___xor__ = [](const Tensor & self, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__xor__(other);
      };
      return wrap(dispatch___xor__(_r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _adaptive_avg_pool2d
static PyObject * THPVariable__adaptive_avg_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_adaptive_avg_pool2d(Tensor input, IntArrayRef[2] output_size)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
  auto dispatch__adaptive_avg_pool2d = [](const Tensor & self, IntArrayRef output_size) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_adaptive_avg_pool2d(self, output_size);
  };
  return wrap(dispatch__adaptive_avg_pool2d(_r.tensor(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _addr
static PyObject * THPVariable__addr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_addr(Tensor input, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(5)) {
    // aten::_addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
    auto dispatch__addr = [](const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_addr(self, vec1, vec2, beta, alpha);
    };
    return wrap(dispatch__addr(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
  } else {
    // aten::_addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
    auto dispatch__addr_out = [](Tensor out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_addr_out(out, self, vec1, vec2, beta, alpha);
    };
    return wrap(dispatch__addr_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _addr_
static PyObject * THPVariable__addr_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_addr_(Tensor input, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
  auto dispatch__addr_ = [](Tensor self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_addr_(self, vec1, vec2, beta, alpha);
  };
  return wrap(dispatch__addr_(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _baddbmm_mkl_
static PyObject * THPVariable__baddbmm_mkl_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_baddbmm_mkl_(Tensor input, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_baddbmm_mkl_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
  auto dispatch__baddbmm_mkl_ = [](Tensor self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_baddbmm_mkl_(self, batch1, batch2, beta, alpha);
  };
  return wrap(dispatch__baddbmm_mkl_(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _batch_norm_impl_index
static PyObject * THPVariable__batch_norm_impl_index(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double momentum, double eps, bool cudnn_enabled)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)
  auto dispatch__batch_norm_impl_index = [](const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) -> std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t> {
    pybind11::gil_scoped_release no_gil;
    return at::_batch_norm_impl_index(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  };
  return wrap(dispatch__batch_norm_impl_index(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cast_Byte
static PyObject * THPVariable__cast_Byte(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Byte(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cast_Byte(Tensor self, bool non_blocking=False) -> Tensor
  auto dispatch__cast_Byte = [](const Tensor & self, bool non_blocking) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Byte(self, non_blocking);
  };
  return wrap(dispatch__cast_Byte(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cast_Char
static PyObject * THPVariable__cast_Char(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Char(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cast_Char(Tensor self, bool non_blocking=False) -> Tensor
  auto dispatch__cast_Char = [](const Tensor & self, bool non_blocking) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Char(self, non_blocking);
  };
  return wrap(dispatch__cast_Char(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cast_Double
static PyObject * THPVariable__cast_Double(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Double(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cast_Double(Tensor self, bool non_blocking=False) -> Tensor
  auto dispatch__cast_Double = [](const Tensor & self, bool non_blocking) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Double(self, non_blocking);
  };
  return wrap(dispatch__cast_Double(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cast_Float
static PyObject * THPVariable__cast_Float(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Float(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cast_Float(Tensor self, bool non_blocking=False) -> Tensor
  auto dispatch__cast_Float = [](const Tensor & self, bool non_blocking) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Float(self, non_blocking);
  };
  return wrap(dispatch__cast_Float(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cast_Half
static PyObject * THPVariable__cast_Half(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Half(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cast_Half(Tensor self, bool non_blocking=False) -> Tensor
  auto dispatch__cast_Half = [](const Tensor & self, bool non_blocking) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Half(self, non_blocking);
  };
  return wrap(dispatch__cast_Half(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cast_Int
static PyObject * THPVariable__cast_Int(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Int(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cast_Int(Tensor self, bool non_blocking=False) -> Tensor
  auto dispatch__cast_Int = [](const Tensor & self, bool non_blocking) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Int(self, non_blocking);
  };
  return wrap(dispatch__cast_Int(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cast_Long
static PyObject * THPVariable__cast_Long(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Long(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cast_Long(Tensor self, bool non_blocking=False) -> Tensor
  auto dispatch__cast_Long = [](const Tensor & self, bool non_blocking) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Long(self, non_blocking);
  };
  return wrap(dispatch__cast_Long(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cast_Short
static PyObject * THPVariable__cast_Short(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Short(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cast_Short(Tensor self, bool non_blocking=False) -> Tensor
  auto dispatch__cast_Short = [](const Tensor & self, bool non_blocking) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Short(self, non_blocking);
  };
  return wrap(dispatch__cast_Short(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cat
static PyObject * THPVariable__cat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cat(TensorList tensors, int64_t dim=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::_cat(Tensor[] tensors, int dim=0) -> Tensor
    auto dispatch__cat = [](TensorList tensors, int64_t dim) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_cat(tensors, dim);
    };
    return wrap(dispatch__cat(_r.tensorlist(0), _r.toInt64(1)));
  } else {
    // aten::_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch__cat_out = [](Tensor out, TensorList tensors, int64_t dim) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_cat_out(out, tensors, dim);
    };
    return wrap(dispatch__cat_out(_r.tensor(2), _r.tensorlist(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _convolution
static PyObject * THPVariable__convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_convolution(Tensor input, Tensor weight, Tensor? bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled)",
  }, /*traceable=*/true);

  ParsedArgs<12> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor
  auto dispatch__convolution = [](const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
  };
  return wrap(dispatch__convolution(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toBool(6), _r.intlist(7), _r.toInt64(8), _r.toBool(9), _r.toBool(10), _r.toBool(11)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _convolution_nogroup
static PyObject * THPVariable__convolution_nogroup(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding) -> Tensor
  auto dispatch__convolution_nogroup = [](const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_convolution_nogroup(input, weight, bias, stride, padding, dilation, transposed, output_padding);
  };
  return wrap(dispatch__convolution_nogroup(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toBool(6), _r.intlist(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _copy_from
static PyObject * THPVariable__copy_from(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_copy_from(Tensor input, Tensor dst, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
  auto dispatch__copy_from = [](const Tensor & self, const Tensor & dst, bool non_blocking) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_copy_from(self, dst, non_blocking);
  };
  return wrap(dispatch__copy_from(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _ctc_loss
static PyObject * THPVariable__ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_ctc_loss(Tensor log_probs, Tensor targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank=0, bool zero_infinity=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
  auto dispatch__ctc_loss = [](const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool zero_infinity) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
  };
  return wrap(dispatch__ctc_loss(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3), _r.toInt64(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cudnn_ctc_loss
static PyObject * THPVariable__cudnn_ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_ctc_loss(Tensor log_probs, Tensor targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)
  auto dispatch__cudnn_ctc_loss = [](const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
  };
  return wrap(dispatch__cudnn_ctc_loss(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3), _r.toInt64(4), _r.toBool(5), _r.toBool(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cudnn_init_dropout_state
static PyObject * THPVariable__cudnn_init_dropout_state(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cudnn_init_dropout_state(float dropout, bool train, int dropout_seed, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartype(3))
      .device(_r.device(5))
      .layout(_r.layout(4).layout)
      .requires_grad(_r.toBool(7))
      .pinned_memory(_r.toBool(6));
  torch::utils::maybe_initialize_cuda(options);
  auto dispatch__cudnn_init_dropout_state = [](double dropout, bool train, int64_t dropout_seed, const TensorOptions & options) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::_cudnn_init_dropout_state(dropout, train, dropout_seed, options);
  };
  return wrap(dispatch__cudnn_init_dropout_state(_r.toDouble(0), _r.toBool(1), _r.toInt64(2), options));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cudnn_rnn
static PyObject * THPVariable__cudnn_rnn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_rnn(Tensor input, TensorList weight, int64_t weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, Tensor? dropout_state)",
  }, /*traceable=*/true);

  ParsedArgs<15> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
  auto dispatch__cudnn_rnn = [](const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state) -> std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_cudnn_rnn(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
  };
  return wrap(dispatch__cudnn_rnn(_r.tensor(0), _r.tensorlist(1), _r.toInt64(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.toInt64(6), _r.toInt64(7), _r.toInt64(8), _r.toBool(9), _r.toDouble(10), _r.toBool(11), _r.toBool(12), _r.intlist(13), _r.tensor(14)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cudnn_rnn_flatten_weight
static PyObject * THPVariable__cudnn_rnn_flatten_weight(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cudnn_rnn_flatten_weight(Tensor[] weight_arr, int weight_stride0, int input_size, int mode, int hidden_size, int num_layers, bool batch_first, bool bidirectional) -> Tensor
  auto dispatch__cudnn_rnn_flatten_weight = [](TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cudnn_rnn_flatten_weight(weight_arr, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
  };
  return wrap(dispatch__cudnn_rnn_flatten_weight(_r.tensorlist(0), _r.toInt64(1), _r.toInt64(2), _r.toInt64(3), _r.toInt64(4), _r.toInt64(5), _r.toBool(6), _r.toBool(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cufft_clear_plan_cache
static PyObject * THPVariable__cufft_clear_plan_cache(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cufft_clear_plan_cache(int64_t device_index)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cufft_clear_plan_cache(int device_index) -> ()
  auto dispatch__cufft_clear_plan_cache = [](int64_t device_index) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_cufft_clear_plan_cache(device_index);
  };
  dispatch__cufft_clear_plan_cache(_r.toInt64(0));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cufft_get_plan_cache_max_size
static PyObject * THPVariable__cufft_get_plan_cache_max_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cufft_get_plan_cache_max_size(int64_t device_index)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cufft_get_plan_cache_max_size(int device_index) -> int
  auto dispatch__cufft_get_plan_cache_max_size = [](int64_t device_index) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return at::_cufft_get_plan_cache_max_size(device_index);
  };
  return wrap(dispatch__cufft_get_plan_cache_max_size(_r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cufft_get_plan_cache_size
static PyObject * THPVariable__cufft_get_plan_cache_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cufft_get_plan_cache_size(int64_t device_index)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cufft_get_plan_cache_size(int device_index) -> int
  auto dispatch__cufft_get_plan_cache_size = [](int64_t device_index) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return at::_cufft_get_plan_cache_size(device_index);
  };
  return wrap(dispatch__cufft_get_plan_cache_size(_r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cufft_set_plan_cache_max_size
static PyObject * THPVariable__cufft_set_plan_cache_max_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cufft_set_plan_cache_max_size(int64_t device_index, int64_t max_size)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_cufft_set_plan_cache_max_size(int device_index, int max_size) -> ()
  auto dispatch__cufft_set_plan_cache_max_size = [](int64_t device_index, int64_t max_size) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_cufft_set_plan_cache_max_size(device_index, max_size);
  };
  dispatch__cufft_set_plan_cache_max_size(_r.toInt64(0), _r.toInt64(1));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _debug_has_internal_overlap
static PyObject * THPVariable__debug_has_internal_overlap(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_debug_has_internal_overlap(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_debug_has_internal_overlap(Tensor self) -> int
  auto dispatch__debug_has_internal_overlap = [](const Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return at::_debug_has_internal_overlap(self);
  };
  return wrap(dispatch__debug_has_internal_overlap(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _dim_arange
static PyObject * THPVariable__dim_arange(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_dim_arange(Tensor like, int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_dim_arange(Tensor like, int dim) -> Tensor
  auto dispatch__dim_arange = [](const Tensor & like, int64_t dim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_dim_arange(like, dim);
  };
  return wrap(dispatch__dim_arange(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _dirichlet_grad
static PyObject * THPVariable__dirichlet_grad(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_dirichlet_grad(Tensor x, Tensor alpha, Tensor total)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_dirichlet_grad(Tensor x, Tensor alpha, Tensor total) -> Tensor
  auto dispatch__dirichlet_grad = [](const Tensor & x, const Tensor & alpha, const Tensor & total) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_dirichlet_grad(x, alpha, total);
  };
  return wrap(dispatch__dirichlet_grad(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _embedding_bag
static PyObject * THPVariable__embedding_bag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int64_t mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)
  auto dispatch__embedding_bag = [](const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights, bool include_last_offset) -> std::tuple<Tensor,Tensor,Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
  };
  return wrap(dispatch__embedding_bag(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3), _r.toInt64(4), _r.toBool(5), _r.tensor(6), _r.toBool(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _empty_affine_quantized
static PyObject * THPVariable__empty_affine_quantized(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_empty_affine_quantized(IntArrayRef size, *, double scale=1, int64_t zero_point=0, MemoryFormat? memory_format=MemoryFormat::Contiguous, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_empty_affine_quantized(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartype(4))
      .device(_r.device(6))
      .layout(_r.layout(5).layout)
      .requires_grad(_r.toBool(8))
      .pinned_memory(_r.toBool(7));
  torch::utils::maybe_initialize_cuda(options);
  auto dispatch__empty_affine_quantized = [](IntArrayRef size, const TensorOptions & options, double scale, int64_t zero_point, c10::optional<MemoryFormat> memory_format) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::_empty_affine_quantized(size, options, scale, zero_point, memory_format);
  };
  return wrap(dispatch__empty_affine_quantized(_r.intlist(0), options, _r.toDouble(1), _r.toInt64(2), _r.memoryformatOptional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _empty_per_channel_affine_quantized
static PyObject * THPVariable__empty_per_channel_affine_quantized(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_empty_per_channel_affine_quantized(IntArrayRef size, *, Tensor scales, Tensor zero_points, int64_t axis, MemoryFormat? memory_format=MemoryFormat::Contiguous, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_empty_per_channel_affine_quantized(int[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartype(5))
      .device(_r.device(7))
      .layout(_r.layout(6).layout)
      .requires_grad(_r.toBool(9))
      .pinned_memory(_r.toBool(8));
  torch::utils::maybe_initialize_cuda(options);
  auto dispatch__empty_per_channel_affine_quantized = [](IntArrayRef size, const Tensor & scales, const Tensor & zero_points, int64_t axis, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::_empty_per_channel_affine_quantized(size, scales, zero_points, axis, options, memory_format);
  };
  return wrap(dispatch__empty_per_channel_affine_quantized(_r.intlist(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), options, _r.memoryformatOptional(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _fft_with_size
static PyObject * THPVariable__fft_with_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fft_with_size(Tensor input, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntArrayRef checked_signal_sizes, bool normalized, bool onesided, IntArrayRef output_sizes)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_fft_with_size(Tensor self, int signal_ndim, bool complex_input, bool complex_output, bool inverse, int[] checked_signal_sizes, bool normalized, bool onesided, int[] output_sizes) -> Tensor
  auto dispatch__fft_with_size = [](const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntArrayRef checked_signal_sizes, bool normalized, bool onesided, IntArrayRef output_sizes) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_fft_with_size(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
  };
  return wrap(dispatch__fft_with_size(_r.tensor(0), _r.toInt64(1), _r.toBool(2), _r.toBool(3), _r.toBool(4), _r.intlist(5), _r.toBool(6), _r.toBool(7), _r.intlist(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _fused_dropout
static PyObject * THPVariable__fused_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fused_dropout(Tensor input, double p, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)
  auto dispatch__fused_dropout = [](const Tensor & self, double p, Generator * generator) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_fused_dropout(self, p, generator);
  };
  return wrap(dispatch__fused_dropout(_r.tensor(0), _r.toDouble(1), _r.generator(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _has_compatible_shallow_copy_type
static PyObject * THPVariable__has_compatible_shallow_copy_type(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_has_compatible_shallow_copy_type(Tensor input, Tensor from)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_has_compatible_shallow_copy_type(Tensor self, Tensor from) -> bool
  auto dispatch__has_compatible_shallow_copy_type = [](const Tensor & self, const Tensor & from) -> bool {
    pybind11::gil_scoped_release no_gil;
    return at::_has_compatible_shallow_copy_type(self, from);
  };
  return wrap(dispatch__has_compatible_shallow_copy_type(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _index_copy_
static PyObject * THPVariable__index_copy_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_index_copy_(Tensor input, int64_t dim, Tensor index, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
  auto dispatch__index_copy_ = [](Tensor self, int64_t dim, const Tensor & index, const Tensor & source) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_index_copy_(self, dim, index, source);
  };
  return wrap(dispatch__index_copy_(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _index_put_impl_
static PyObject * THPVariable__index_put_impl_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_index_put_impl_(Tensor input, TensorList? indices, Tensor values, bool accumulate=False, bool unsafe=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)
  auto dispatch__index_put_impl_ = [](Tensor self, TensorList indices, const Tensor & values, bool accumulate, bool unsafe) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_index_put_impl_(self, indices, values, accumulate, unsafe);
  };
  return wrap(dispatch__index_put_impl_(_r.tensor(0), _r.tensorlist(1), _r.tensor(2), _r.toBool(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _log_softmax
static PyObject * THPVariable__log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_log_softmax(Tensor input, int64_t dim, bool half_to_float)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
  auto dispatch__log_softmax = [](const Tensor & self, int64_t dim, bool half_to_float) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_log_softmax(self, dim, half_to_float);
  };
  return wrap(dispatch__log_softmax(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _log_softmax_backward_data
static PyObject * THPVariable__log_softmax_backward_data(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_log_softmax_backward_data(Tensor grad_output, Tensor output, int64_t dim, Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
  auto dispatch__log_softmax_backward_data = [](const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_log_softmax_backward_data(grad_output, output, dim, self);
  };
  return wrap(dispatch__log_softmax_backward_data(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.tensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _lu_solve_helper
static PyObject * THPVariable__lu_solve_helper(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_lu_solve_helper(Tensor input, Tensor LU_data, Tensor LU_pivots)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_lu_solve_helper(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor
  auto dispatch__lu_solve_helper = [](const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_lu_solve_helper(self, LU_data, LU_pivots);
  };
  return wrap(dispatch__lu_solve_helper(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _lu_with_info
static PyObject * THPVariable__lu_with_info(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_lu_with_info(Tensor input, bool pivot=True, bool check_errors=True)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_lu_with_info(Tensor self, bool pivot=True, bool check_errors=True) -> (Tensor, Tensor, Tensor)
  auto dispatch__lu_with_info = [](const Tensor & self, bool pivot, bool check_errors) -> std::tuple<Tensor,Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_lu_with_info(self, pivot, check_errors);
  };
  return wrap(dispatch__lu_with_info(_r.tensor(0), _r.toBool(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _make_per_channel_quantized_tensor
static PyObject * THPVariable__make_per_channel_quantized_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_make_per_channel_quantized_tensor(Tensor input, Tensor scale, Tensor zero_point, int64_t axis)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_make_per_channel_quantized_tensor(Tensor self, Tensor scale, Tensor zero_point, int axis) -> Tensor
  auto dispatch__make_per_channel_quantized_tensor = [](const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_make_per_channel_quantized_tensor(self, scale, zero_point, axis);
  };
  return wrap(dispatch__make_per_channel_quantized_tensor(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _make_per_tensor_quantized_tensor
static PyObject * THPVariable__make_per_tensor_quantized_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_make_per_tensor_quantized_tensor(Tensor input, double scale, int64_t zero_point)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_make_per_tensor_quantized_tensor(Tensor self, float scale, int zero_point) -> Tensor
  auto dispatch__make_per_tensor_quantized_tensor = [](const Tensor & self, double scale, int64_t zero_point) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_make_per_tensor_quantized_tensor(self, scale, zero_point);
  };
  return wrap(dispatch__make_per_tensor_quantized_tensor(_r.tensor(0), _r.toDouble(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _masked_scale
static PyObject * THPVariable__masked_scale(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_masked_scale(Tensor input, Tensor mask, double scale)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_masked_scale(Tensor self, Tensor mask, float scale) -> Tensor
  auto dispatch__masked_scale = [](const Tensor & self, const Tensor & mask, double scale) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_masked_scale(self, mask, scale);
  };
  return wrap(dispatch__masked_scale(_r.tensor(0), _r.tensor(1), _r.toDouble(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _max
static PyObject * THPVariable__max(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_max(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(3)) {
    // aten::_max(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)
    auto dispatch__max = [](const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_max(self, dim, keepdim);
    };
    return wrap(dispatch__max(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  } else {
    // aten::_max.max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(3);
    auto dispatch__max_out = [](Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_max_out(max, max_indices, self, dim, keepdim);
    };
    return wrap(dispatch__max_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _min
static PyObject * THPVariable__min(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_min(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(3)) {
    // aten::_min(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)
    auto dispatch__min = [](const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_min(self, dim, keepdim);
    };
    return wrap(dispatch__min(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  } else {
    // aten::_min.min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(3);
    auto dispatch__min_out = [](Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_min_out(min, min_indices, self, dim, keepdim);
    };
    return wrap(dispatch__min_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _mkldnn_reshape
static PyObject * THPVariable__mkldnn_reshape(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_mkldnn_reshape(Tensor input, IntArrayRef shape)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_mkldnn_reshape(Tensor self, int[] shape) -> Tensor
  auto dispatch__mkldnn_reshape = [](const Tensor & self, IntArrayRef shape) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_mkldnn_reshape(self, shape);
  };
  return wrap(dispatch__mkldnn_reshape(_r.tensor(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _mkldnn_transpose
static PyObject * THPVariable__mkldnn_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_mkldnn_transpose(Tensor input, int64_t dim0, int64_t dim1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_mkldnn_transpose(Tensor self, int dim0, int dim1) -> Tensor
  auto dispatch__mkldnn_transpose = [](const Tensor & self, int64_t dim0, int64_t dim1) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_mkldnn_transpose(self, dim0, dim1);
  };
  return wrap(dispatch__mkldnn_transpose(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _mkldnn_transpose_
static PyObject * THPVariable__mkldnn_transpose_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_mkldnn_transpose_(Tensor input, int64_t dim0, int64_t dim1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_mkldnn_transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
  auto dispatch__mkldnn_transpose_ = [](Tensor self, int64_t dim0, int64_t dim1) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_mkldnn_transpose_(self, dim0, dim1);
  };
  return wrap(dispatch__mkldnn_transpose_(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _mode
static PyObject * THPVariable__mode(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_mode(Tensor input, int64_t dim=-1, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(3)) {
    // aten::_mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor, Tensor)
    auto dispatch__mode = [](const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_mode(self, dim, keepdim);
    };
    return wrap(dispatch__mode(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  } else {
    // aten::_mode.values(Tensor self, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(3);
    auto dispatch__mode_out = [](Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_mode_out(values, indices, self, dim, keepdim);
    };
    return wrap(dispatch__mode_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _multinomial_alias_draw
static PyObject * THPVariable__multinomial_alias_draw(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_multinomial_alias_draw(Tensor J, Tensor q, int64_t num_samples, *, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_multinomial_alias_draw(Tensor J, Tensor q, int num_samples, *, Generator? generator=None) -> Tensor
  auto dispatch__multinomial_alias_draw = [](const Tensor & J, const Tensor & q, int64_t num_samples, Generator * generator) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_multinomial_alias_draw(J, q, num_samples, generator);
  };
  return wrap(dispatch__multinomial_alias_draw(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.generator(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _multinomial_alias_setup
static PyObject * THPVariable__multinomial_alias_setup(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_multinomial_alias_setup(Tensor probs)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_multinomial_alias_setup(Tensor probs) -> (Tensor, Tensor)
  auto dispatch__multinomial_alias_setup = [](const Tensor & probs) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_multinomial_alias_setup(probs);
  };
  return wrap(dispatch__multinomial_alias_setup(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nnpack_available
static PyObject * THPVariable__nnpack_available(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  // aten::_nnpack_available() -> bool
  auto dispatch__nnpack_available = []() -> bool {
    pybind11::gil_scoped_release no_gil;
    return at::_nnpack_available();
  };
  return wrap(dispatch__nnpack_available());
  END_HANDLE_TH_ERRORS
}

// _nnpack_spatial_convolution
static PyObject * THPVariable__nnpack_spatial_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nnpack_spatial_convolution(Tensor input, Tensor weight, Tensor? bias, IntArrayRef[2] padding, IntArrayRef[2] stride=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_nnpack_spatial_convolution(Tensor input, Tensor weight, Tensor? bias, int[2] padding, int[2] stride=1) -> Tensor
  auto dispatch__nnpack_spatial_convolution = [](const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nnpack_spatial_convolution(input, weight, bias, padding, stride);
  };
  return wrap(dispatch__nnpack_spatial_convolution(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _pack_padded_sequence
static PyObject * THPVariable__pack_padded_sequence(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)
  auto dispatch__pack_padded_sequence = [](const Tensor & input, const Tensor & lengths, bool batch_first) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_pack_padded_sequence(input, lengths, batch_first);
  };
  return wrap(dispatch__pack_padded_sequence(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _pad_packed_sequence
static PyObject * THPVariable__pad_packed_sequence(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int64_t total_length)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int total_length) -> (Tensor, Tensor)
  auto dispatch__pad_packed_sequence = [](const Tensor & data, const Tensor & batch_sizes, bool batch_first, Scalar padding_value, int64_t total_length) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length);
  };
  return wrap(dispatch__pad_packed_sequence(_r.tensor(0), _r.tensor(1), _r.toBool(2), _r.scalar(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _reshape_from_tensor
static PyObject * THPVariable__reshape_from_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_reshape_from_tensor(Tensor input, Tensor shape)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_reshape_from_tensor(Tensor self, Tensor shape) -> Tensor
  auto dispatch__reshape_from_tensor = [](const Tensor & self, const Tensor & shape) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_reshape_from_tensor(self, shape);
  };
  return wrap(dispatch__reshape_from_tensor(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _s_where
static PyObject * THPVariable__s_where(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_s_where(Tensor condition, Tensor input, Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_s_where(Tensor condition, Tensor self, Tensor other) -> Tensor
  auto dispatch__s_where = [](const Tensor & condition, const Tensor & self, const Tensor & other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_s_where(condition, self, other);
  };
  return wrap(dispatch__s_where(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sample_dirichlet
static PyObject * THPVariable__sample_dirichlet(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sample_dirichlet(Tensor input, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_sample_dirichlet(Tensor self, Generator? generator=None) -> Tensor
  auto dispatch__sample_dirichlet = [](const Tensor & self, Generator * generator) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sample_dirichlet(self, generator);
  };
  return wrap(dispatch__sample_dirichlet(_r.tensor(0), _r.generator(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _shape_as_tensor
static PyObject * THPVariable__shape_as_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_shape_as_tensor(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_shape_as_tensor(Tensor self) -> Tensor
  auto dispatch__shape_as_tensor = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_shape_as_tensor(self);
  };
  return wrap(dispatch__shape_as_tensor(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sobol_engine_draw
static PyObject * THPVariable__sobol_engine_draw(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sobol_engine_draw(Tensor quasi, int64_t n, Tensor sobolstate, int64_t dimension, int64_t num_generated, ScalarType? dtype)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_sobol_engine_draw(Tensor quasi, int n, Tensor sobolstate, int dimension, int num_generated, ScalarType? dtype) -> (Tensor, Tensor)
  auto dispatch__sobol_engine_draw = [](const Tensor & quasi, int64_t n, const Tensor & sobolstate, int64_t dimension, int64_t num_generated, c10::optional<ScalarType> dtype) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_sobol_engine_draw(quasi, n, sobolstate, dimension, num_generated, dtype);
  };
  return wrap(dispatch__sobol_engine_draw(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4), _r.scalartypeOptional(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sobol_engine_ff_
static PyObject * THPVariable__sobol_engine_ff_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sobol_engine_ff_(Tensor input, int64_t n, Tensor sobolstate, int64_t dimension, int64_t num_generated)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_sobol_engine_ff_(Tensor(a!) self, int n, Tensor sobolstate, int dimension, int num_generated) -> Tensor(a!)
  auto dispatch__sobol_engine_ff_ = [](Tensor self, int64_t n, const Tensor & sobolstate, int64_t dimension, int64_t num_generated) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sobol_engine_ff_(self, n, sobolstate, dimension, num_generated);
  };
  return wrap(dispatch__sobol_engine_ff_(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sobol_engine_initialize_state_
static PyObject * THPVariable__sobol_engine_initialize_state_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sobol_engine_initialize_state_(Tensor input, int64_t dimension)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_sobol_engine_initialize_state_(Tensor(a!) self, int dimension) -> Tensor(a!)
  auto dispatch__sobol_engine_initialize_state_ = [](Tensor self, int64_t dimension) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sobol_engine_initialize_state_(self, dimension);
  };
  return wrap(dispatch__sobol_engine_initialize_state_(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sobol_engine_scramble_
static PyObject * THPVariable__sobol_engine_scramble_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sobol_engine_scramble_(Tensor input, Tensor ltm, int64_t dimension)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_sobol_engine_scramble_(Tensor(a!) self, Tensor ltm, int dimension) -> Tensor(a!)
  auto dispatch__sobol_engine_scramble_ = [](Tensor self, const Tensor & ltm, int64_t dimension) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sobol_engine_scramble_(self, ltm, dimension);
  };
  return wrap(dispatch__sobol_engine_scramble_(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _softmax
static PyObject * THPVariable__softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_softmax(Tensor input, int64_t dim, bool half_to_float)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
  auto dispatch__softmax = [](const Tensor & self, int64_t dim, bool half_to_float) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_softmax(self, dim, half_to_float);
  };
  return wrap(dispatch__softmax(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _softmax_backward_data
static PyObject * THPVariable__softmax_backward_data(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_softmax_backward_data(Tensor grad_output, Tensor output, int64_t dim, Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
  auto dispatch__softmax_backward_data = [](const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_softmax_backward_data(grad_output, output, dim, self);
  };
  return wrap(dispatch__softmax_backward_data(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.tensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_addmm
static PyObject * THPVariable__sparse_addmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_addmm(Tensor input, Tensor sparse, Tensor dense, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_sparse_addmm(Tensor self, Tensor sparse, Tensor dense, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  auto dispatch__sparse_addmm = [](const Tensor & self, const Tensor & sparse, const Tensor & dense, Scalar beta, Scalar alpha) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_addmm(self, sparse, dense, beta, alpha);
  };
  return wrap(dispatch__sparse_addmm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_mm
static PyObject * THPVariable__sparse_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_mm(Tensor sparse, Tensor dense)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_sparse_mm(Tensor sparse, Tensor dense) -> Tensor
  auto dispatch__sparse_mm = [](const Tensor & sparse, const Tensor & dense) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_mm(sparse, dense);
  };
  return wrap(dispatch__sparse_mm(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _sparse_sum
static PyObject * THPVariable__sparse_sum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_sum(Tensor input)",
    "_sparse_sum(Tensor input, *, ScalarType dtype)",
    "_sparse_sum(Tensor input, IntArrayRef[1] dim)",
    "_sparse_sum(Tensor input, IntArrayRef[1] dim, *, ScalarType dtype)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::_sparse_sum(Tensor self) -> Tensor
      auto dispatch__sparse_sum = [](const Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_sum(self);
      };
      return wrap(dispatch__sparse_sum(_r.tensor(0)));
    }
    case 1: {
      // aten::_sparse_sum.dtype(Tensor self, *, ScalarType dtype) -> Tensor
      auto dispatch__sparse_sum = [](const Tensor & self, ScalarType dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_sum(self, dtype);
      };
      return wrap(dispatch__sparse_sum(_r.tensor(0), _r.scalartype(1)));
    }
    case 2: {
      // aten::_sparse_sum.dim(Tensor self, int[1] dim) -> Tensor
      auto dispatch__sparse_sum = [](const Tensor & self, IntArrayRef dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_sum(self, dim);
      };
      return wrap(dispatch__sparse_sum(_r.tensor(0), _r.intlist(1)));
    }
    case 3: {
      // aten::_sparse_sum.dim_dtype(Tensor self, int[1] dim, *, ScalarType dtype) -> Tensor
      auto dispatch__sparse_sum = [](const Tensor & self, IntArrayRef dim, ScalarType dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_sum(self, dim, dtype);
      };
      return wrap(dispatch__sparse_sum(_r.tensor(0), _r.intlist(1), _r.scalartype(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _standard_gamma
static PyObject * THPVariable__standard_gamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_standard_gamma(Tensor input, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_standard_gamma(Tensor self, Generator? generator=None) -> Tensor
  auto dispatch__standard_gamma = [](const Tensor & self, Generator * generator) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_standard_gamma(self, generator);
  };
  return wrap(dispatch__standard_gamma(_r.tensor(0), _r.generator(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _standard_gamma_grad
static PyObject * THPVariable__standard_gamma_grad(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_standard_gamma_grad(Tensor input, Tensor output)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_standard_gamma_grad(Tensor self, Tensor output) -> Tensor
  auto dispatch__standard_gamma_grad = [](const Tensor & self, const Tensor & output) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_standard_gamma_grad(self, output);
  };
  return wrap(dispatch__standard_gamma_grad(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _std
static PyObject * THPVariable__std(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_std(Tensor input, bool unbiased=True)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_std(Tensor self, bool unbiased=True) -> Tensor
  auto dispatch__std = [](const Tensor & self, bool unbiased) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_std(self, unbiased);
  };
  return wrap(dispatch__std(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _trilinear
static PyObject * THPVariable__trilinear(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_trilinear(Tensor i1, Tensor i2, Tensor i3, IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3, IntArrayRef sumdim, int64_t unroll_dim=1)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> Tensor
  auto dispatch__trilinear = [](const Tensor & i1, const Tensor & i2, const Tensor & i3, IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3, IntArrayRef sumdim, int64_t unroll_dim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
  };
  return wrap(dispatch__trilinear(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.intlist(6), _r.toInt64(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _unique
static PyObject * THPVariable__unique(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_unique(Tensor input, bool sorted=True, bool return_inverse=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_unique(Tensor self, bool sorted=True, bool return_inverse=False) -> (Tensor, Tensor)
  auto dispatch__unique = [](const Tensor & self, bool sorted, bool return_inverse) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_unique(self, sorted, return_inverse);
  };
  return wrap(dispatch__unique(_r.tensor(0), _r.toBool(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _unique2
static PyObject * THPVariable__unique2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_unique2(Tensor input, bool sorted=True, bool return_inverse=False, bool return_counts=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
  auto dispatch__unique2 = [](const Tensor & self, bool sorted, bool return_inverse, bool return_counts) -> std::tuple<Tensor,Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_unique2(self, sorted, return_inverse, return_counts);
  };
  return wrap(dispatch__unique2(_r.tensor(0), _r.toBool(1), _r.toBool(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _use_cudnn_ctc_loss
static PyObject * THPVariable__use_cudnn_ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_use_cudnn_ctc_loss(Tensor log_probs, Tensor targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank)",
  }, /*traceable=*/false);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_use_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank) -> bool
  auto dispatch__use_cudnn_ctc_loss = [](const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank) -> bool {
    pybind11::gil_scoped_release no_gil;
    return at::_use_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
  };
  return wrap(dispatch__use_cudnn_ctc_loss(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _var
static PyObject * THPVariable__var(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_var(Tensor input, bool unbiased=True)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_var(Tensor self, bool unbiased=True) -> Tensor
  auto dispatch__var = [](const Tensor & self, bool unbiased) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_var(self, unbiased);
  };
  return wrap(dispatch__var(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _weight_norm
static PyObject * THPVariable__weight_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_weight_norm(Tensor v, Tensor g, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor
  auto dispatch__weight_norm = [](const Tensor & v, const Tensor & g, int64_t dim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_weight_norm(v, g, dim);
  };
  return wrap(dispatch__weight_norm(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _weight_norm_cuda_interface
static PyObject * THPVariable__weight_norm_cuda_interface(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_weight_norm_cuda_interface(Tensor v, Tensor g, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::_weight_norm_cuda_interface(Tensor v, Tensor g, int dim=0) -> (Tensor, Tensor)
  auto dispatch__weight_norm_cuda_interface = [](const Tensor & v, const Tensor & g, int64_t dim) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_weight_norm_cuda_interface(v, g, dim);
  };
  return wrap(dispatch__weight_norm_cuda_interface(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// abs
static PyObject * THPVariable_abs(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "abs(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::abs(Tensor self) -> Tensor
    auto dispatch_abs = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.abs();
    };
    return wrap(dispatch_abs(_r.tensor(0)));
  } else {
    // aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_abs_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::abs_out(out, self);
    };
    return wrap(dispatch_abs_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// abs_
static PyObject * THPVariable_abs_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "abs_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::abs_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_abs_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.abs_();
  };
  return wrap(dispatch_abs_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// acos
static PyObject * THPVariable_acos(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "acos(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::acos(Tensor self) -> Tensor
    auto dispatch_acos = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.acos();
    };
    return wrap(dispatch_acos(_r.tensor(0)));
  } else {
    // aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_acos_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::acos_out(out, self);
    };
    return wrap(dispatch_acos_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// acos_
static PyObject * THPVariable_acos_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "acos_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::acos_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_acos_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.acos_();
  };
  return wrap(dispatch_acos_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// adaptive_avg_pool1d
static PyObject * THPVariable_adaptive_avg_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_avg_pool1d(Tensor input, IntArrayRef[1] output_size)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor
  auto dispatch_adaptive_avg_pool1d = [](const Tensor & self, IntArrayRef output_size) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::adaptive_avg_pool1d(self, output_size);
  };
  return wrap(dispatch_adaptive_avg_pool1d(_r.tensor(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// adaptive_max_pool1d
static PyObject * THPVariable_adaptive_max_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_max_pool1d(Tensor input, IntArrayRef[1] output_size)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)
  auto dispatch_adaptive_max_pool1d = [](const Tensor & self, IntArrayRef output_size) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::adaptive_max_pool1d(self, output_size);
  };
  return wrap(dispatch_adaptive_max_pool1d(_r.tensor(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// add
static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "add(Tensor input, Scalar alpha, Tensor other, *, Tensor out=None)|deprecated",
    "add(Tensor input, Tensor other, *, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // [deprecated] aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
        auto dispatch_add = [](const Tensor & self, Scalar alpha, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.add(other, alpha);
        };
        return wrap(dispatch_add(_r.tensor(0), _r.scalar(1), _r.tensor(2)));
      } else {
        // [deprecated] aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_add_out = [](const Tensor & self, Scalar alpha, const Tensor & other, Tensor out) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::add_out(out, self, other, alpha);
        };
        return wrap(dispatch_add_out(_r.tensor(0), _r.scalar(1), _r.tensor(2), _r.tensor(3)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
        auto dispatch_add = [](const Tensor & self, const Tensor & other, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.add(other, alpha);
        };
        return wrap(dispatch_add(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
      } else {
        // aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_add_out = [](Tensor out, const Tensor & self, const Tensor & other, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::add_out(out, self, other, alpha);
        };
        return wrap(dispatch_add_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.scalar(2)));
      }
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
  static PythonArgParser parser({
    "addbmm(Scalar beta, Tensor input, Scalar alpha, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "addbmm(Scalar beta, Tensor input, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "addbmm(Tensor input, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // [deprecated] aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_addbmm = [](Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addbmm(batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_addbmm(_r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      } else {
        // [deprecated] aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addbmm_out = [](Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor out) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addbmm_out(out, self, batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_addbmm_out(_r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4), _r.tensor(5)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // [deprecated] aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_addbmm = [](Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addbmm(batch1, batch2, beta, 1);
        };
        return wrap(dispatch_addbmm(_r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // [deprecated] aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addbmm_out = [](Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor out) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addbmm_out(out, self, batch1, batch2, beta, 1);
        };
        return wrap(dispatch_addbmm_out(_r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4)));
      }
    }
    case 2: {
      if (_r.isNone(5)) {
        // aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_addbmm = [](const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addbmm(batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_addbmm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      } else {
        // aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addbmm_out = [](Tensor out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addbmm_out(out, self, batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_addbmm_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      }
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
  static PythonArgParser parser({
    "addcdiv(Tensor input, Scalar value, Tensor tensor1, Tensor tensor2, *, Tensor out=None)|deprecated",
    "addcdiv(Tensor input, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // [deprecated] aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
        auto dispatch_addcdiv = [](const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addcdiv(tensor1, tensor2, value);
        };
        return wrap(dispatch_addcdiv(_r.tensor(0), _r.scalar(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // [deprecated] aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addcdiv_out = [](const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor out) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addcdiv_out(out, self, tensor1, tensor2, value);
        };
        return wrap(dispatch_addcdiv_out(_r.tensor(0), _r.scalar(1), _r.tensor(2), _r.tensor(3), _r.tensor(4)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
        auto dispatch_addcdiv = [](const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addcdiv(tensor1, tensor2, value);
        };
        return wrap(dispatch_addcdiv(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3)));
      } else {
        // aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addcdiv_out = [](Tensor out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addcdiv_out(out, self, tensor1, tensor2, value);
        };
        return wrap(dispatch_addcdiv_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3)));
      }
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
  static PythonArgParser parser({
    "addcmul(Tensor input, Scalar value, Tensor tensor1, Tensor tensor2, *, Tensor out=None)|deprecated",
    "addcmul(Tensor input, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // [deprecated] aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
        auto dispatch_addcmul = [](const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addcmul(tensor1, tensor2, value);
        };
        return wrap(dispatch_addcmul(_r.tensor(0), _r.scalar(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // [deprecated] aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addcmul_out = [](const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor out) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addcmul_out(out, self, tensor1, tensor2, value);
        };
        return wrap(dispatch_addcmul_out(_r.tensor(0), _r.scalar(1), _r.tensor(2), _r.tensor(3), _r.tensor(4)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
        auto dispatch_addcmul = [](const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addcmul(tensor1, tensor2, value);
        };
        return wrap(dispatch_addcmul(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3)));
      } else {
        // aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addcmul_out = [](Tensor out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addcmul_out(out, self, tensor1, tensor2, value);
        };
        return wrap(dispatch_addcmul_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3)));
      }
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
  static PythonArgParser parser({
    "addmm(Scalar beta, Tensor input, Scalar alpha, Tensor mat1, Tensor mat2, *, Tensor out=None)|deprecated",
    "addmm(Scalar beta, Tensor input, Tensor mat1, Tensor mat2, *, Tensor out=None)|deprecated",
    "addmm(Tensor input, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // [deprecated] aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_addmm = [](Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addmm(mat1, mat2, beta, alpha);
        };
        return wrap(dispatch_addmm(_r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      } else {
        // [deprecated] aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addmm_out = [](Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2, Tensor out) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addmm_out(out, self, mat1, mat2, beta, alpha);
        };
        return wrap(dispatch_addmm_out(_r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4), _r.tensor(5)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // [deprecated] aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_addmm = [](Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addmm(mat1, mat2, beta, 1);
        };
        return wrap(dispatch_addmm(_r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // [deprecated] aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addmm_out = [](Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Tensor out) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addmm_out(out, self, mat1, mat2, beta, 1);
        };
        return wrap(dispatch_addmm_out(_r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4)));
      }
    }
    case 2: {
      if (_r.isNone(5)) {
        // aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_addmm = [](const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addmm(mat1, mat2, beta, alpha);
        };
        return wrap(dispatch_addmm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      } else {
        // aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addmm_out = [](Tensor out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addmm_out(out, self, mat1, mat2, beta, alpha);
        };
        return wrap(dispatch_addmm_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      }
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
  static PythonArgParser parser({
    "addmv(Scalar beta, Tensor input, Scalar alpha, Tensor mat, Tensor vec, *, Tensor out=None)|deprecated",
    "addmv(Scalar beta, Tensor input, Tensor mat, Tensor vec, *, Tensor out=None)|deprecated",
    "addmv(Tensor input, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // [deprecated] aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_addmv = [](Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addmv(mat, vec, beta, alpha);
        };
        return wrap(dispatch_addmv(_r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      } else {
        // [deprecated] aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addmv_out = [](Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec, Tensor out) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addmv_out(out, self, mat, vec, beta, alpha);
        };
        return wrap(dispatch_addmv_out(_r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4), _r.tensor(5)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // [deprecated] aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_addmv = [](Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addmv(mat, vec, beta, 1);
        };
        return wrap(dispatch_addmv(_r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // [deprecated] aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addmv_out = [](Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec, Tensor out) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addmv_out(out, self, mat, vec, beta, 1);
        };
        return wrap(dispatch_addmv_out(_r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4)));
      }
    }
    case 2: {
      if (_r.isNone(5)) {
        // aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_addmv = [](const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addmv(mat, vec, beta, alpha);
        };
        return wrap(dispatch_addmv(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      } else {
        // aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addmv_out = [](Tensor out, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addmv_out(out, self, mat, vec, beta, alpha);
        };
        return wrap(dispatch_addmv_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      }
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
  static PythonArgParser parser({
    "addmv_(Scalar beta, Tensor input, Scalar alpha, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Scalar beta, Tensor input, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Tensor input, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addmv_ = [](Scalar beta, Tensor self, Scalar alpha, const Tensor & mat, const Tensor & vec) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv_(mat, vec, beta, alpha);
      };
      return wrap(dispatch_addmv_(_r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
    }
    case 1: {
      // [deprecated] aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addmv_ = [](Scalar beta, Tensor self, const Tensor & mat, const Tensor & vec) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv_(mat, vec, beta, 1);
      };
      return wrap(dispatch_addmv_(_r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
    }
    case 2: {
      // aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      auto dispatch_addmv_ = [](Tensor self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv_(mat, vec, beta, alpha);
      };
      return wrap(dispatch_addmv_(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
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
  static PythonArgParser parser({
    "addr(Scalar beta, Tensor input, Scalar alpha, Tensor vec1, Tensor vec2, *, Tensor out=None)|deprecated",
    "addr(Scalar beta, Tensor input, Tensor vec1, Tensor vec2, *, Tensor out=None)|deprecated",
    "addr(Tensor input, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // [deprecated] aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_addr = [](Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addr(vec1, vec2, beta, alpha);
        };
        return wrap(dispatch_addr(_r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      } else {
        // [deprecated] aten::addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addr_out = [](Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2, Tensor out) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addr_out(out, self, vec1, vec2, beta, alpha);
        };
        return wrap(dispatch_addr_out(_r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4), _r.tensor(5)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // [deprecated] aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_addr = [](Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addr(vec1, vec2, beta, 1);
        };
        return wrap(dispatch_addr(_r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // [deprecated] aten::addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addr_out = [](Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Tensor out) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addr_out(out, self, vec1, vec2, beta, 1);
        };
        return wrap(dispatch_addr_out(_r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4)));
      }
    }
    case 2: {
      if (_r.isNone(5)) {
        // aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_addr = [](const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addr(vec1, vec2, beta, alpha);
        };
        return wrap(dispatch_addr(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      } else {
        // aten::addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_addr_out = [](Tensor out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addr_out(out, self, vec1, vec2, beta, alpha);
        };
        return wrap(dispatch_addr_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// affine_grid_generator
static PyObject * THPVariable_affine_grid_generator(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "affine_grid_generator(Tensor theta, IntArrayRef size, bool align_corners)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::affine_grid_generator(Tensor theta, int[] size, bool align_corners) -> Tensor
  auto dispatch_affine_grid_generator = [](const Tensor & theta, IntArrayRef size, bool align_corners) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::affine_grid_generator(theta, size, align_corners);
  };
  return wrap(dispatch_affine_grid_generator(_r.tensor(0), _r.intlist(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// align_tensors
static PyObject * THPVariable_align_tensors(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "align_tensors(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::align_tensors(Tensor[] tensors) -> Tensor[]
  auto dispatch_align_tensors = [](TensorList tensors) -> std::vector<Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::align_tensors(tensors);
  };
  return wrap(dispatch_align_tensors(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// all
static PyObject * THPVariable_all(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "all(Tensor input)",
    "all(Tensor input, Dimname dim, bool keepdim=False, *, Tensor out=None)",
    "all(Tensor input, int64_t dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::all(Tensor self) -> Tensor
      auto dispatch_all = [](const Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.all();
      };
      return wrap(dispatch_all(_r.tensor(0)));
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::all.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
        auto dispatch_all = [](const Tensor & self, Dimname dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.all(dim, keepdim);
        };
        return wrap(dispatch_all(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::all.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_all_out = [](Tensor out, const Tensor & self, Dimname dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::all_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_all_out(_r.tensor(3), _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
        auto dispatch_all = [](const Tensor & self, int64_t dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.all(dim, keepdim);
        };
        return wrap(dispatch_all(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_all_out = [](Tensor out, const Tensor & self, int64_t dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::all_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_all_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// allclose
static PyObject * THPVariable_allclose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "allclose(Tensor input, Tensor other, double rtol=1e-05, double atol=1e-08, bool equal_nan=False)",
  }, /*traceable=*/false);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool
  auto dispatch_allclose = [](const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.allclose(other, rtol, atol, equal_nan);
  };
  return wrap(dispatch_allclose(_r.tensor(0), _r.tensor(1), _r.toDouble(2), _r.toDouble(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// alpha_dropout
static PyObject * THPVariable_alpha_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "alpha_dropout(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor
  auto dispatch_alpha_dropout = [](const Tensor & input, double p, bool train) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::alpha_dropout(input, p, train);
  };
  return wrap(dispatch_alpha_dropout(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// alpha_dropout_
static PyObject * THPVariable_alpha_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "alpha_dropout_(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
  auto dispatch_alpha_dropout_ = [](Tensor self, double p, bool train) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::alpha_dropout_(self, p, train);
  };
  return wrap(dispatch_alpha_dropout_(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// angle
static PyObject * THPVariable_angle(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "angle(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::angle(Tensor self) -> Tensor
    auto dispatch_angle = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.angle();
    };
    return wrap(dispatch_angle(_r.tensor(0)));
  } else {
    // aten::angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_angle_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::angle_out(out, self);
    };
    return wrap(dispatch_angle_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// any
static PyObject * THPVariable_any(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "any(Tensor input)",
    "any(Tensor input, Dimname dim, bool keepdim=False, *, Tensor out=None)",
    "any(Tensor input, int64_t dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::any(Tensor self) -> Tensor
      auto dispatch_any = [](const Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.any();
      };
      return wrap(dispatch_any(_r.tensor(0)));
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::any.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
        auto dispatch_any = [](const Tensor & self, Dimname dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.any(dim, keepdim);
        };
        return wrap(dispatch_any(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::any.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_any_out = [](Tensor out, const Tensor & self, Dimname dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::any_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_any_out(_r.tensor(3), _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
        auto dispatch_any = [](const Tensor & self, int64_t dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.any(dim, keepdim);
        };
        return wrap(dispatch_any(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_any_out = [](Tensor out, const Tensor & self, int64_t dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::any_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_any_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// argmax
static PyObject * THPVariable_argmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "argmax(Tensor input, int64_t? dim=None, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
  auto dispatch_argmax = [](const Tensor & self, c10::optional<int64_t> dim, bool keepdim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.argmax(dim, keepdim);
  };
  return wrap(dispatch_argmax(_r.tensor(0), _r.toInt64Optional(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// argmin
static PyObject * THPVariable_argmin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "argmin(Tensor input, int64_t? dim=None, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
  auto dispatch_argmin = [](const Tensor & self, c10::optional<int64_t> dim, bool keepdim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.argmin(dim, keepdim);
  };
  return wrap(dispatch_argmin(_r.tensor(0), _r.toInt64Optional(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// argsort
static PyObject * THPVariable_argsort(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "argsort(Tensor input, Dimname dim, bool descending=False)",
    "argsort(Tensor input, int64_t dim=-1, bool descending=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::argsort.dimname(Tensor self, Dimname dim, bool descending=False) -> Tensor
      auto dispatch_argsort = [](const Tensor & self, Dimname dim, bool descending) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.argsort(dim, descending);
      };
      return wrap(dispatch_argsort(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
    }
    case 1: {
      // aten::argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor
      auto dispatch_argsort = [](const Tensor & self, int64_t dim, bool descending) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.argsort(dim, descending);
      };
      return wrap(dispatch_argsort(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// as_strided
static PyObject * THPVariable_as_strided(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "as_strided(Tensor input, IntArrayRef size, IntArrayRef stride, int64_t? storage_offset=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)
  auto dispatch_as_strided = [](const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.as_strided(size, stride, storage_offset);
  };
  return wrap(dispatch_as_strided(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.toInt64Optional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// as_strided_
static PyObject * THPVariable_as_strided_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "as_strided_(Tensor input, IntArrayRef size, IntArrayRef stride, int64_t? storage_offset=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::as_strided_(Tensor(a!) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a!)
  auto dispatch_as_strided_ = [](Tensor self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.as_strided_(size, stride, storage_offset);
  };
  return wrap(dispatch_as_strided_(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.toInt64Optional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// asin
static PyObject * THPVariable_asin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "asin(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::asin(Tensor self) -> Tensor
    auto dispatch_asin = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.asin();
    };
    return wrap(dispatch_asin(_r.tensor(0)));
  } else {
    // aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_asin_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::asin_out(out, self);
    };
    return wrap(dispatch_asin_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// asin_
static PyObject * THPVariable_asin_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "asin_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::asin_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_asin_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.asin_();
  };
  return wrap(dispatch_asin_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// atan
static PyObject * THPVariable_atan(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::atan(Tensor self) -> Tensor
    auto dispatch_atan = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.atan();
    };
    return wrap(dispatch_atan(_r.tensor(0)));
  } else {
    // aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_atan_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::atan_out(out, self);
    };
    return wrap(dispatch_atan_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// atan2
static PyObject * THPVariable_atan2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan2(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::atan2(Tensor self, Tensor other) -> Tensor
    auto dispatch_atan2 = [](const Tensor & self, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.atan2(other);
    };
    return wrap(dispatch_atan2(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_atan2_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::atan2_out(out, self, other);
    };
    return wrap(dispatch_atan2_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// atan_
static PyObject * THPVariable_atan_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::atan_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_atan_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.atan_();
  };
  return wrap(dispatch_atan_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// avg_pool1d
static PyObject * THPVariable_avg_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "avg_pool1d(Tensor input, IntArrayRef[1] kernel_size, IntArrayRef[1] stride=None, IntArrayRef[1] padding=0, bool ceil_mode=False, bool count_include_pad=True)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor
  auto dispatch_avg_pool1d = [](const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::avg_pool1d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  };
  return wrap(dispatch_avg_pool1d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.toBool(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// baddbmm
static PyObject * THPVariable_baddbmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "baddbmm(Scalar beta, Tensor input, Scalar alpha, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "baddbmm(Scalar beta, Tensor input, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "baddbmm(Tensor input, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // [deprecated] aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_baddbmm = [](Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.baddbmm(batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_baddbmm(_r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      } else {
        // [deprecated] aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_baddbmm_out = [](Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor out) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::baddbmm_out(out, self, batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_baddbmm_out(_r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4), _r.tensor(5)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // [deprecated] aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_baddbmm = [](Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.baddbmm(batch1, batch2, beta, 1);
        };
        return wrap(dispatch_baddbmm(_r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // [deprecated] aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_baddbmm_out = [](Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor out) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::baddbmm_out(out, self, batch1, batch2, beta, 1);
        };
        return wrap(dispatch_baddbmm_out(_r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4)));
      }
    }
    case 2: {
      if (_r.isNone(5)) {
        // aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_baddbmm = [](const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.baddbmm(batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_baddbmm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      } else {
        // aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_baddbmm_out = [](Tensor out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::baddbmm_out(out, self, batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_baddbmm_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bartlett_window
static PyObject * THPVariable_bartlett_window(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bartlett_window(int64_t window_length, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "bartlett_window(int64_t window_length, bool periodic, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartype(1))
          .device(_r.device(3))
          .layout(_r.layout(2).layout)
          .requires_grad(_r.toBool(5))
          .pinned_memory(_r.toBool(4));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_bartlett_window = [](int64_t window_length, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::bartlett_window(window_length, options);
      };
      return wrap(dispatch_bartlett_window(_r.toInt64(0), options));
    }
    case 1: {
      // aten::bartlett_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartype(2))
          .device(_r.device(4))
          .layout(_r.layout(3).layout)
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_bartlett_window = [](int64_t window_length, bool periodic, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::bartlett_window(window_length, periodic, options);
      };
      return wrap(dispatch_bartlett_window(_r.toInt64(0), _r.toBool(1), options));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm
static PyObject * THPVariable_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double momentum, double eps, bool cudnn_enabled)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
  auto dispatch_batch_norm = [](const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  };
  return wrap(dispatch_batch_norm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm_backward_elemt
static PyObject * THPVariable_batch_norm_backward_elemt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor mean_dy, Tensor mean_dy_xmu)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor mean_dy, Tensor mean_dy_xmu) -> Tensor
  auto dispatch_batch_norm_backward_elemt = [](const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & weight, const Tensor & mean_dy, const Tensor & mean_dy_xmu) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu);
  };
  return wrap(dispatch_batch_norm_backward_elemt(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.tensor(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm_backward_reduce
static PyObject * THPVariable_batch_norm_backward_reduce(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g) -> (Tensor, Tensor, Tensor, Tensor)
  auto dispatch_batch_norm_backward_reduce = [](const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & weight, bool input_g, bool weight_g, bool bias_g) -> std::tuple<Tensor,Tensor,Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::batch_norm_backward_reduce(grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
  };
  return wrap(dispatch_batch_norm_backward_reduce(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.toBool(5), _r.toBool(6), _r.toBool(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm_elemt
static PyObject * THPVariable_batch_norm_elemt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, double eps, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(6)) {
    // aten::batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps) -> Tensor
    auto dispatch_batch_norm_elemt = [](const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & invstd, double eps) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::batch_norm_elemt(input, weight, bias, mean, invstd, eps);
    };
    return wrap(dispatch_batch_norm_elemt(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.toDouble(5)));
  } else {
    // aten::batch_norm_elemt.out(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_batch_norm_elemt_out = [](Tensor out, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & invstd, double eps) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::batch_norm_elemt_out(out, input, weight, bias, mean, invstd, eps);
    };
    return wrap(dispatch_batch_norm_elemt_out(_r.tensor(6), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.toDouble(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm_gather_stats
static PyObject * THPVariable_batch_norm_gather_stats(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, double momentum, double eps, int64_t count)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count) -> (Tensor, Tensor)
  auto dispatch_batch_norm_gather_stats = [](const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & running_mean, const Tensor & running_var, double momentum, double eps, int64_t count) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::batch_norm_gather_stats(input, mean, invstd, running_mean, running_var, momentum, eps, count);
  };
  return wrap(dispatch_batch_norm_gather_stats(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.toDouble(5), _r.toDouble(6), _r.toInt64(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm_gather_stats_with_counts
static PyObject * THPVariable_batch_norm_gather_stats_with_counts(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, double momentum, double eps, IntArrayRef counts)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int[] counts) -> (Tensor, Tensor)
  auto dispatch_batch_norm_gather_stats_with_counts = [](const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & running_mean, const Tensor & running_var, double momentum, double eps, IntArrayRef counts) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
  };
  return wrap(dispatch_batch_norm_gather_stats_with_counts(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.toDouble(5), _r.toDouble(6), _r.intlist(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm_stats
static PyObject * THPVariable_batch_norm_stats(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm_stats(Tensor input, double eps)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)
  auto dispatch_batch_norm_stats = [](const Tensor & input, double eps) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::batch_norm_stats(input, eps);
  };
  return wrap(dispatch_batch_norm_stats(_r.tensor(0), _r.toDouble(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm_update_stats
static PyObject * THPVariable_batch_norm_update_stats(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, double momentum)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum) -> (Tensor, Tensor)
  auto dispatch_batch_norm_update_stats = [](const Tensor & input, const Tensor & running_mean, const Tensor & running_var, double momentum) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::batch_norm_update_stats(input, running_mean, running_var, momentum);
  };
  return wrap(dispatch_batch_norm_update_stats(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toDouble(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bernoulli
static PyObject * THPVariable_bernoulli(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bernoulli(Tensor input, *, Generator generator=None, Tensor out=None)",
    "bernoulli(Tensor input, double p, *, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor
        auto dispatch_bernoulli = [](const Tensor & self, Generator * generator) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bernoulli(generator);
        };
        return wrap(dispatch_bernoulli(_r.tensor(0), _r.generator(1)));
      } else {
        // aten::bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_bernoulli_out = [](Tensor out, const Tensor & self, Generator * generator) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bernoulli_out(out, self, generator);
        };
        return wrap(dispatch_bernoulli_out(_r.tensor(2), _r.tensor(0), _r.generator(1)));
      }
    }
    case 1: {
      // aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor
      auto dispatch_bernoulli = [](const Tensor & self, double p, Generator * generator) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bernoulli(p, generator);
      };
      return wrap(dispatch_bernoulli(_r.tensor(0), _r.toDouble(1), _r.generator(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// bilinear
static PyObject * THPVariable_bilinear(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor
  auto dispatch_bilinear = [](const Tensor & input1, const Tensor & input2, const Tensor & weight, const Tensor & bias) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::bilinear(input1, input2, weight, bias);
  };
  return wrap(dispatch_bilinear(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// binary_cross_entropy_with_logits
static PyObject * THPVariable_binary_cross_entropy_with_logits(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "binary_cross_entropy_with_logits(Tensor input, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int64_t reduction=at::Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor
  auto dispatch_binary_cross_entropy_with_logits = [](const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction);
  };
  return wrap(dispatch_binary_cross_entropy_with_logits(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// bincount
static PyObject * THPVariable_bincount(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bincount(Tensor input, Tensor? weights=None, int64_t minlength=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor
  auto dispatch_bincount = [](const Tensor & self, const Tensor & weights, int64_t minlength) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.bincount(weights, minlength);
  };
  return wrap(dispatch_bincount(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_and
static PyObject * THPVariable_bitwise_and(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bitwise_and(Tensor input, Tensor other, *, Tensor out=None)",
    "bitwise_and(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
        auto dispatch_bitwise_and = [](const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_and(other);
        };
        return wrap(dispatch_bitwise_and(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_bitwise_and_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_and_out(out, self, other);
        };
        return wrap(dispatch_bitwise_and_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
        auto dispatch_bitwise_and = [](const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_and(other);
        };
        return wrap(dispatch_bitwise_and(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::bitwise_and.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_bitwise_and_out = [](Tensor out, const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_and_out(out, self, other);
        };
        return wrap(dispatch_bitwise_and_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// bitwise_not
static PyObject * THPVariable_bitwise_not(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bitwise_not(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::bitwise_not(Tensor self) -> Tensor
    auto dispatch_bitwise_not = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.bitwise_not();
    };
    return wrap(dispatch_bitwise_not(_r.tensor(0)));
  } else {
    // aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_bitwise_not_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::bitwise_not_out(out, self);
    };
    return wrap(dispatch_bitwise_not_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_or
static PyObject * THPVariable_bitwise_or(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bitwise_or(Tensor input, Tensor other, *, Tensor out=None)",
    "bitwise_or(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor
        auto dispatch_bitwise_or = [](const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_or(other);
        };
        return wrap(dispatch_bitwise_or(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::bitwise_or.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_bitwise_or_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_or_out(out, self, other);
        };
        return wrap(dispatch_bitwise_or_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor
        auto dispatch_bitwise_or = [](const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_or(other);
        };
        return wrap(dispatch_bitwise_or(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::bitwise_or.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_bitwise_or_out = [](Tensor out, const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_or_out(out, self, other);
        };
        return wrap(dispatch_bitwise_or_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
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
  static PythonArgParser parser({
    "bitwise_xor(Tensor input, Tensor other, *, Tensor out=None)",
    "bitwise_xor(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor
        auto dispatch_bitwise_xor = [](const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_xor(other);
        };
        return wrap(dispatch_bitwise_xor(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::bitwise_xor.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_bitwise_xor_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_xor_out(out, self, other);
        };
        return wrap(dispatch_bitwise_xor_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor
        auto dispatch_bitwise_xor = [](const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_xor(other);
        };
        return wrap(dispatch_bitwise_xor(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::bitwise_xor.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_bitwise_xor_out = [](Tensor out, const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_xor_out(out, self, other);
        };
        return wrap(dispatch_bitwise_xor_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// blackman_window
static PyObject * THPVariable_blackman_window(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "blackman_window(int64_t window_length, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "blackman_window(int64_t window_length, bool periodic, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartype(1))
          .device(_r.device(3))
          .layout(_r.layout(2).layout)
          .requires_grad(_r.toBool(5))
          .pinned_memory(_r.toBool(4));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_blackman_window = [](int64_t window_length, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::blackman_window(window_length, options);
      };
      return wrap(dispatch_blackman_window(_r.toInt64(0), options));
    }
    case 1: {
      // aten::blackman_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartype(2))
          .device(_r.device(4))
          .layout(_r.layout(3).layout)
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_blackman_window = [](int64_t window_length, bool periodic, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::blackman_window(window_length, periodic, options);
      };
      return wrap(dispatch_blackman_window(_r.toInt64(0), _r.toBool(1), options));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// bmm
static PyObject * THPVariable_bmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bmm(Tensor input, Tensor mat2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::bmm(Tensor self, Tensor mat2) -> Tensor
    auto dispatch_bmm = [](const Tensor & self, const Tensor & mat2) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.bmm(mat2);
    };
    return wrap(dispatch_bmm(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_bmm_out = [](Tensor out, const Tensor & self, const Tensor & mat2) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::bmm_out(out, self, mat2);
    };
    return wrap(dispatch_bmm_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// broadcast_tensors
static PyObject * THPVariable_broadcast_tensors(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "broadcast_tensors(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::broadcast_tensors(Tensor[] tensors) -> Tensor[]
  auto dispatch_broadcast_tensors = [](TensorList tensors) -> std::vector<Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::broadcast_tensors(tensors);
  };
  return wrap(dispatch_broadcast_tensors(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// can_cast
static PyObject * THPVariable_can_cast(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "can_cast(ScalarType from, ScalarType to)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::can_cast(ScalarType from, ScalarType to) -> bool
  auto dispatch_can_cast = [](ScalarType from, ScalarType to) -> bool {
    pybind11::gil_scoped_release no_gil;
    return at::can_cast(from, to);
  };
  return wrap(dispatch_can_cast(_r.scalartype(0), _r.scalartype(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cartesian_prod
static PyObject * THPVariable_cartesian_prod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cartesian_prod(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::cartesian_prod(Tensor[] tensors) -> Tensor
  auto dispatch_cartesian_prod = [](TensorList tensors) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cartesian_prod(tensors);
  };
  return wrap(dispatch_cartesian_prod(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cat
static PyObject * THPVariable_cat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cat(TensorList tensors, Dimname dim, *, Tensor out=None)",
    "cat(TensorList tensors, int64_t dim=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::cat.names(Tensor[] tensors, Dimname dim) -> Tensor
        auto dispatch_cat = [](TensorList tensors, Dimname dim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cat(tensors, dim);
        };
        return wrap(dispatch_cat(_r.tensorlist(0), _r.dimname(1)));
      } else {
        // aten::cat.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_cat_out = [](Tensor out, TensorList tensors, Dimname dim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cat_out(out, tensors, dim);
        };
        return wrap(dispatch_cat_out(_r.tensor(2), _r.tensorlist(0), _r.dimname(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::cat(Tensor[] tensors, int dim=0) -> Tensor
        auto dispatch_cat = [](TensorList tensors, int64_t dim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cat(tensors, dim);
        };
        return wrap(dispatch_cat(_r.tensorlist(0), _r.toInt64(1)));
      } else {
        // aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_cat_out = [](Tensor out, TensorList tensors, int64_t dim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cat_out(out, tensors, dim);
        };
        return wrap(dispatch_cat_out(_r.tensor(2), _r.tensorlist(0), _r.toInt64(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cdist
static PyObject * THPVariable_cdist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cdist(Tensor x1, Tensor x2, double p=2, int64_t? compute_mode=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::cdist(Tensor x1, Tensor x2, float p=2, int? compute_mode=None) -> Tensor
  auto dispatch_cdist = [](const Tensor & x1, const Tensor & x2, double p, c10::optional<int64_t> compute_mode) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cdist(x1, x2, p, compute_mode);
  };
  return wrap(dispatch_cdist(_r.tensor(0), _r.tensor(1), _r.toDouble(2), _r.toInt64Optional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ceil
static PyObject * THPVariable_ceil(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ceil(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::ceil(Tensor self) -> Tensor
    auto dispatch_ceil = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.ceil();
    };
    return wrap(dispatch_ceil(_r.tensor(0)));
  } else {
    // aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_ceil_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::ceil_out(out, self);
    };
    return wrap(dispatch_ceil_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ceil_
static PyObject * THPVariable_ceil_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ceil_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::ceil_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_ceil_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ceil_();
  };
  return wrap(dispatch_ceil_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// celu
static PyObject * THPVariable_celu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "celu(Tensor input, Scalar alpha=1.0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::celu(Tensor self, Scalar alpha=1.0) -> Tensor
  auto dispatch_celu = [](const Tensor & self, Scalar alpha) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::celu(self, alpha);
  };
  return wrap(dispatch_celu(_r.tensor(0), _r.scalar(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// celu_
static PyObject * THPVariable_celu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "celu_(Tensor input, Scalar alpha=1.0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::celu_(Tensor(a!) self, Scalar alpha=1.0) -> Tensor(a!)
  auto dispatch_celu_ = [](Tensor self, Scalar alpha) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::celu_(self, alpha);
  };
  return wrap(dispatch_celu_(_r.tensor(0), _r.scalar(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// chain_matmul
static PyObject * THPVariable_chain_matmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "chain_matmul(TensorList matrices)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::chain_matmul(Tensor[] matrices) -> Tensor
  auto dispatch_chain_matmul = [](TensorList matrices) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::chain_matmul(matrices);
  };
  return wrap(dispatch_chain_matmul(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cholesky
static PyObject * THPVariable_cholesky(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cholesky(Tensor input, bool upper=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::cholesky(Tensor self, bool upper=False) -> Tensor
    auto dispatch_cholesky = [](const Tensor & self, bool upper) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.cholesky(upper);
    };
    return wrap(dispatch_cholesky(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::cholesky.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_cholesky_out = [](Tensor out, const Tensor & self, bool upper) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::cholesky_out(out, self, upper);
    };
    return wrap(dispatch_cholesky_out(_r.tensor(2), _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cholesky_inverse
static PyObject * THPVariable_cholesky_inverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cholesky_inverse(Tensor input, bool upper=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor
    auto dispatch_cholesky_inverse = [](const Tensor & self, bool upper) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.cholesky_inverse(upper);
    };
    return wrap(dispatch_cholesky_inverse(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::cholesky_inverse.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_cholesky_inverse_out = [](Tensor out, const Tensor & self, bool upper) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::cholesky_inverse_out(out, self, upper);
    };
    return wrap(dispatch_cholesky_inverse_out(_r.tensor(2), _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cholesky_solve
static PyObject * THPVariable_cholesky_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cholesky_solve(Tensor input, Tensor input2, bool upper=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(3)) {
    // aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor
    auto dispatch_cholesky_solve = [](const Tensor & self, const Tensor & input2, bool upper) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.cholesky_solve(input2, upper);
    };
    return wrap(dispatch_cholesky_solve(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
  } else {
    // aten::cholesky_solve.out(Tensor self, Tensor input2, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_cholesky_solve_out = [](Tensor out, const Tensor & self, const Tensor & input2, bool upper) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::cholesky_solve_out(out, self, input2, upper);
    };
    return wrap(dispatch_cholesky_solve_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// chunk
static PyObject * THPVariable_chunk(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "chunk(Tensor input, int64_t chunks, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]
  auto dispatch_chunk = [](const Tensor & self, int64_t chunks, int64_t dim) -> std::vector<Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.chunk(chunks, dim);
  };
  return wrap(dispatch_chunk(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clamp
static PyObject * THPVariable_clamp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp(Tensor input, Scalar? min=None, Scalar? max=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(3)) {
    // aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
    auto dispatch_clamp = [](const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.clamp(min, max);
    };
    return wrap(dispatch_clamp(_r.tensor(0), _r.scalarOptional(1), _r.scalarOptional(2)));
  } else {
    // aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_clamp_out = [](Tensor out, const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::clamp_out(out, self, min, max);
    };
    return wrap(dispatch_clamp_out(_r.tensor(3), _r.tensor(0), _r.scalarOptional(1), _r.scalarOptional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clamp_
static PyObject * THPVariable_clamp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_(Tensor input, Scalar? min=None, Scalar? max=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
  auto dispatch_clamp_ = [](Tensor self, c10::optional<Scalar> min, c10::optional<Scalar> max) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.clamp_(min, max);
  };
  return wrap(dispatch_clamp_(_r.tensor(0), _r.scalarOptional(1), _r.scalarOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clamp_max
static PyObject * THPVariable_clamp_max(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_max(Tensor input, Scalar max, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::clamp_max(Tensor self, Scalar max) -> Tensor
    auto dispatch_clamp_max = [](const Tensor & self, Scalar max) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.clamp_max(max);
    };
    return wrap(dispatch_clamp_max(_r.tensor(0), _r.scalar(1)));
  } else {
    // aten::clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_clamp_max_out = [](Tensor out, const Tensor & self, Scalar max) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::clamp_max_out(out, self, max);
    };
    return wrap(dispatch_clamp_max_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clamp_max_
static PyObject * THPVariable_clamp_max_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_max_(Tensor input, Scalar max)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)
  auto dispatch_clamp_max_ = [](Tensor self, Scalar max) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.clamp_max_(max);
  };
  return wrap(dispatch_clamp_max_(_r.tensor(0), _r.scalar(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clamp_min
static PyObject * THPVariable_clamp_min(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_min(Tensor input, Scalar min, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::clamp_min(Tensor self, Scalar min) -> Tensor
    auto dispatch_clamp_min = [](const Tensor & self, Scalar min) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.clamp_min(min);
    };
    return wrap(dispatch_clamp_min(_r.tensor(0), _r.scalar(1)));
  } else {
    // aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_clamp_min_out = [](Tensor out, const Tensor & self, Scalar min) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::clamp_min_out(out, self, min);
    };
    return wrap(dispatch_clamp_min_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clamp_min_
static PyObject * THPVariable_clamp_min_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_min_(Tensor input, Scalar min)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)
  auto dispatch_clamp_min_ = [](Tensor self, Scalar min) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.clamp_min_(min);
  };
  return wrap(dispatch_clamp_min_(_r.tensor(0), _r.scalar(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clone
static PyObject * THPVariable_clone(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clone(Tensor input, *, MemoryFormat? memory_format=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
  auto dispatch_clone = [](const Tensor & self, c10::optional<MemoryFormat> memory_format) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.clone(memory_format);
  };
  return wrap(dispatch_clone(_r.tensor(0), _r.memoryformatOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// combinations
static PyObject * THPVariable_combinations(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "combinations(Tensor input, int64_t r=2, bool with_replacement=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor
  auto dispatch_combinations = [](const Tensor & self, int64_t r, bool with_replacement) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::combinations(self, r, with_replacement);
  };
  return wrap(dispatch_combinations(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conj
static PyObject * THPVariable_conj(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conj(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::conj(Tensor self) -> Tensor
    auto dispatch_conj = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.conj();
    };
    return wrap(dispatch_conj(_r.tensor(0)));
  } else {
    // aten::conj.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_conj_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::conj_out(out, self);
    };
    return wrap(dispatch_conj_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// constant_pad_nd
static PyObject * THPVariable_constant_pad_nd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "constant_pad_nd(Tensor input, IntArrayRef pad, Scalar value=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor
  auto dispatch_constant_pad_nd = [](const Tensor & self, IntArrayRef pad, Scalar value) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::constant_pad_nd(self, pad, value);
  };
  return wrap(dispatch_constant_pad_nd(_r.tensor(0), _r.intlist(1), _r.scalar(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conv1d
static PyObject * THPVariable_conv1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv1d(Tensor input, Tensor weight, Tensor? bias=None, IntArrayRef[1] stride=1, IntArrayRef[1] padding=0, IntArrayRef[1] dilation=1, int64_t groups=1)",
  }, /*traceable=*/false);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor
  auto dispatch_conv1d = [](const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::conv1d(input, weight, bias, stride, padding, dilation, groups);
  };
  return wrap(dispatch_conv1d(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toInt64(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conv2d
static PyObject * THPVariable_conv2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv2d(Tensor input, Tensor weight, Tensor? bias=None, IntArrayRef[2] stride=1, IntArrayRef[2] padding=0, IntArrayRef[2] dilation=1, int64_t groups=1)",
  }, /*traceable=*/false);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor
  auto dispatch_conv2d = [](const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::conv2d(input, weight, bias, stride, padding, dilation, groups);
  };
  return wrap(dispatch_conv2d(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toInt64(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conv3d
static PyObject * THPVariable_conv3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv3d(Tensor input, Tensor weight, Tensor? bias=None, IntArrayRef[3] stride=1, IntArrayRef[3] padding=0, IntArrayRef[3] dilation=1, int64_t groups=1)",
  }, /*traceable=*/false);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor
  auto dispatch_conv3d = [](const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::conv3d(input, weight, bias, stride, padding, dilation, groups);
  };
  return wrap(dispatch_conv3d(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toInt64(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conv_tbc
static PyObject * THPVariable_conv_tbc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_tbc(Tensor input, Tensor weight, Tensor bias, int64_t pad=0)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor
  auto dispatch_conv_tbc = [](const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::conv_tbc(self, weight, bias, pad);
  };
  return wrap(dispatch_conv_tbc(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conv_transpose1d
static PyObject * THPVariable_conv_transpose1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, IntArrayRef[1] stride=1, IntArrayRef[1] padding=0, IntArrayRef[1] output_padding=0, int64_t groups=1, IntArrayRef[1] dilation=1)",
  }, /*traceable=*/false);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] output_padding=0, int groups=1, int[1] dilation=1) -> Tensor
  auto dispatch_conv_transpose1d = [](const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  };
  return wrap(dispatch_conv_transpose1d(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toInt64(6), _r.intlist(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conv_transpose2d
static PyObject * THPVariable_conv_transpose2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_transpose2d(Tensor input, Tensor weight, Tensor? bias=None, IntArrayRef[2] stride=1, IntArrayRef[2] padding=0, IntArrayRef[2] output_padding=0, int64_t groups=1, IntArrayRef[2] dilation=1)",
  }, /*traceable=*/false);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor
  auto dispatch_conv_transpose2d = [](const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  };
  return wrap(dispatch_conv_transpose2d(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toInt64(6), _r.intlist(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conv_transpose3d
static PyObject * THPVariable_conv_transpose3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_transpose3d(Tensor input, Tensor weight, Tensor? bias=None, IntArrayRef[3] stride=1, IntArrayRef[3] padding=0, IntArrayRef[3] output_padding=0, int64_t groups=1, IntArrayRef[3] dilation=1)",
  }, /*traceable=*/false);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor
  auto dispatch_conv_transpose3d = [](const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  };
  return wrap(dispatch_conv_transpose3d(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toInt64(6), _r.intlist(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// convolution
static PyObject * THPVariable_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "convolution(Tensor input, Tensor weight, Tensor? bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups)",
  }, /*traceable=*/false);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor
  auto dispatch_convolution = [](const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  };
  return wrap(dispatch_convolution(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toBool(6), _r.intlist(7), _r.toInt64(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cos
static PyObject * THPVariable_cos(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cos(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::cos(Tensor self) -> Tensor
    auto dispatch_cos = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.cos();
    };
    return wrap(dispatch_cos(_r.tensor(0)));
  } else {
    // aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_cos_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::cos_out(out, self);
    };
    return wrap(dispatch_cos_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cos_
static PyObject * THPVariable_cos_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cos_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::cos_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_cos_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cos_();
  };
  return wrap(dispatch_cos_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cosh
static PyObject * THPVariable_cosh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cosh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::cosh(Tensor self) -> Tensor
    auto dispatch_cosh = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.cosh();
    };
    return wrap(dispatch_cosh(_r.tensor(0)));
  } else {
    // aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_cosh_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::cosh_out(out, self);
    };
    return wrap(dispatch_cosh_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cosh_
static PyObject * THPVariable_cosh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cosh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::cosh_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_cosh_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cosh_();
  };
  return wrap(dispatch_cosh_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cosine_embedding_loss
static PyObject * THPVariable_cosine_embedding_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, double margin=0.0, int64_t reduction=at::Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor
  auto dispatch_cosine_embedding_loss = [](const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cosine_embedding_loss(input1, input2, target, margin, reduction);
  };
  return wrap(dispatch_cosine_embedding_loss(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toDouble(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cosine_similarity
static PyObject * THPVariable_cosine_similarity(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cosine_similarity(Tensor x1, Tensor x2, int64_t dim=1, double eps=1e-08)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::cosine_similarity(Tensor x1, Tensor x2, int dim=1, float eps=1e-08) -> Tensor
  auto dispatch_cosine_similarity = [](const Tensor & x1, const Tensor & x2, int64_t dim, double eps) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cosine_similarity(x1, x2, dim, eps);
  };
  return wrap(dispatch_cosine_similarity(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toDouble(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cross
static PyObject * THPVariable_cross(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cross(Tensor input, Tensor other, int64_t? dim=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(3)) {
    // aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor
    auto dispatch_cross = [](const Tensor & self, const Tensor & other, c10::optional<int64_t> dim) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.cross(other, dim);
    };
    return wrap(dispatch_cross(_r.tensor(0), _r.tensor(1), _r.toInt64Optional(2)));
  } else {
    // aten::cross.out(Tensor self, Tensor other, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_cross_out = [](Tensor out, const Tensor & self, const Tensor & other, c10::optional<int64_t> dim) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::cross_out(out, self, other, dim);
    };
    return wrap(dispatch_cross_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64Optional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ctc_loss
static PyObject * THPVariable_ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ctc_loss(Tensor log_probs, Tensor targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank=0, int64_t reduction=at::Reduction::Mean, bool zero_infinity=False)",
    "ctc_loss(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int64_t blank=0, int64_t reduction=at::Reduction::Mean, bool zero_infinity=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::ctc_loss.IntList(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor
      auto dispatch_ctc_loss = [](const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
      };
      return wrap(dispatch_ctc_loss(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3), _r.toInt64(4), _r.toInt64(5), _r.toBool(6)));
    }
    case 1: {
      // aten::ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor
      auto dispatch_ctc_loss = [](const Tensor & log_probs, const Tensor & targets, const Tensor & input_lengths, const Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
      };
      return wrap(dispatch_ctc_loss(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.toInt64(4), _r.toInt64(5), _r.toBool(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cudnn_affine_grid_generator
static PyObject * THPVariable_cudnn_affine_grid_generator(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_affine_grid_generator(Tensor theta, int64_t N, int64_t C, int64_t H, int64_t W)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::cudnn_affine_grid_generator(Tensor theta, int N, int C, int H, int W) -> Tensor grid
  auto dispatch_cudnn_affine_grid_generator = [](const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cudnn_affine_grid_generator(theta, N, C, H, W);
  };
  return wrap(dispatch_cudnn_affine_grid_generator(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toInt64(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cudnn_batch_norm
static PyObject * THPVariable_cudnn_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double exponential_average_factor, double epsilon)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)
  auto dispatch_cudnn_batch_norm = [](const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) -> std::tuple<Tensor,Tensor,Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::cudnn_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
  };
  return wrap(dispatch_cudnn_batch_norm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cudnn_convolution
static PyObject * THPVariable_cudnn_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_convolution(Tensor input, Tensor weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)",
    "cudnn_convolution(Tensor input, Tensor weight, Tensor? bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::cudnn_convolution(Tensor self, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
      auto dispatch_cudnn_convolution = [](const Tensor & self, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::cudnn_convolution(self, weight, padding, stride, dilation, groups, benchmark, deterministic);
      };
      return wrap(dispatch_cudnn_convolution(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toInt64(5), _r.toBool(6), _r.toBool(7)));
    }
    case 1: {
      // aten::cudnn_convolution.deprecated(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
      auto dispatch_cudnn_convolution = [](const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::cudnn_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
      };
      return wrap(dispatch_cudnn_convolution(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toInt64(6), _r.toBool(7), _r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cudnn_convolution_transpose
static PyObject * THPVariable_cudnn_convolution_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_convolution_transpose(Tensor input, Tensor weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)",
    "cudnn_convolution_transpose(Tensor input, Tensor weight, Tensor? bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::cudnn_convolution_transpose(Tensor self, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
      auto dispatch_cudnn_convolution_transpose = [](const Tensor & self, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::cudnn_convolution_transpose(self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
      };
      return wrap(dispatch_cudnn_convolution_transpose(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toInt64(6), _r.toBool(7), _r.toBool(8)));
    }
    case 1: {
      // aten::cudnn_convolution_transpose.deprecated(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
      auto dispatch_cudnn_convolution_transpose = [](const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::cudnn_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
      };
      return wrap(dispatch_cudnn_convolution_transpose(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.intlist(6), _r.toInt64(7), _r.toBool(8), _r.toBool(9)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cudnn_grid_sampler
static PyObject * THPVariable_cudnn_grid_sampler(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_grid_sampler(Tensor input, Tensor grid)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::cudnn_grid_sampler(Tensor self, Tensor grid) -> Tensor output
  auto dispatch_cudnn_grid_sampler = [](const Tensor & self, const Tensor & grid) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cudnn_grid_sampler(self, grid);
  };
  return wrap(dispatch_cudnn_grid_sampler(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cudnn_is_acceptable
static PyObject * THPVariable_cudnn_is_acceptable(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_is_acceptable(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::cudnn_is_acceptable(Tensor self) -> bool
  auto dispatch_cudnn_is_acceptable = [](const Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return at::cudnn_is_acceptable(self);
  };
  return wrap(dispatch_cudnn_is_acceptable(_r.tensor(0)));
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
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.cummax_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "cummax(Tensor input, Dimname dim, *, TensorList[2] out=None)",
    "cummax(Tensor input, int64_t dim, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::cummax.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
        auto dispatch_cummax = [](const Tensor & self, Dimname dim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.cummax(dim);
        };
        return wrap(&NamedTuple, dispatch_cummax(_r.tensor(0), _r.dimname(1)));
      } else {
        // aten::cummax.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(2);
        auto dispatch_cummax_out = [](Tensor & values, Tensor & indices, const Tensor & self, Dimname dim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::cummax_out(values, indices, self, dim);
        };
        return wrap(&NamedTuple1, dispatch_cummax_out(out[0], out[1], _r.tensor(0), _r.dimname(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
        auto dispatch_cummax = [](const Tensor & self, int64_t dim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.cummax(dim);
        };
        return wrap(&NamedTuple, dispatch_cummax(_r.tensor(0), _r.toInt64(1)));
      } else {
        // aten::cummax.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(2);
        auto dispatch_cummax_out = [](Tensor & values, Tensor & indices, const Tensor & self, int64_t dim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::cummax_out(values, indices, self, dim);
        };
        return wrap(&NamedTuple1, dispatch_cummax_out(out[0], out[1], _r.tensor(0), _r.toInt64(1)));
      }
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
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.cummin_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "cummin(Tensor input, Dimname dim, *, TensorList[2] out=None)",
    "cummin(Tensor input, int64_t dim, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::cummin.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
        auto dispatch_cummin = [](const Tensor & self, Dimname dim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.cummin(dim);
        };
        return wrap(&NamedTuple, dispatch_cummin(_r.tensor(0), _r.dimname(1)));
      } else {
        // aten::cummin.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(2);
        auto dispatch_cummin_out = [](Tensor & values, Tensor & indices, const Tensor & self, Dimname dim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::cummin_out(values, indices, self, dim);
        };
        return wrap(&NamedTuple1, dispatch_cummin_out(out[0], out[1], _r.tensor(0), _r.dimname(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)
        auto dispatch_cummin = [](const Tensor & self, int64_t dim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.cummin(dim);
        };
        return wrap(&NamedTuple, dispatch_cummin(_r.tensor(0), _r.toInt64(1)));
      } else {
        // aten::cummin.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(2);
        auto dispatch_cummin_out = [](Tensor & values, Tensor & indices, const Tensor & self, int64_t dim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::cummin_out(values, indices, self, dim);
        };
        return wrap(&NamedTuple1, dispatch_cummin_out(out[0], out[1], _r.tensor(0), _r.toInt64(1)));
      }
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
  static PythonArgParser parser({
    "cumprod(Tensor input, Dimname dim, *, ScalarType? dtype=None, Tensor out=None)",
    "cumprod(Tensor input, int64_t dim, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::cumprod.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
        auto dispatch_cumprod = [](const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.cumprod(dim, dtype);
        };
        return wrap(dispatch_cumprod(_r.tensor(0), _r.dimname(1), _r.scalartypeOptional(2)));
      } else {
        // aten::cumprod.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_cumprod_out = [](Tensor out, const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cumprod_out(out, self, dim, dtype);
        };
        return wrap(dispatch_cumprod_out(_r.tensor(3), _r.tensor(0), _r.dimname(1), _r.scalartypeOptional(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
        auto dispatch_cumprod = [](const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.cumprod(dim, dtype);
        };
        return wrap(dispatch_cumprod(_r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
      } else {
        // aten::cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_cumprod_out = [](Tensor out, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cumprod_out(out, self, dim, dtype);
        };
        return wrap(dispatch_cumprod_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
      }
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
  static PythonArgParser parser({
    "cumsum(Tensor input, Dimname dim, *, ScalarType? dtype=None, Tensor out=None)",
    "cumsum(Tensor input, int64_t dim, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::cumsum.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
        auto dispatch_cumsum = [](const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.cumsum(dim, dtype);
        };
        return wrap(dispatch_cumsum(_r.tensor(0), _r.dimname(1), _r.scalartypeOptional(2)));
      } else {
        // aten::cumsum.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_cumsum_out = [](Tensor out, const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cumsum_out(out, self, dim, dtype);
        };
        return wrap(dispatch_cumsum_out(_r.tensor(3), _r.tensor(0), _r.dimname(1), _r.scalartypeOptional(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
        auto dispatch_cumsum = [](const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.cumsum(dim, dtype);
        };
        return wrap(dispatch_cumsum(_r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
      } else {
        // aten::cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_cumsum_out = [](Tensor out, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cumsum_out(out, self, dim, dtype);
        };
        return wrap(dispatch_cumsum_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// dequantize
static PyObject * THPVariable_dequantize(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dequantize(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::dequantize(Tensor self) -> Tensor
  auto dispatch_dequantize = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.dequantize();
  };
  return wrap(dispatch_dequantize(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// det
static PyObject * THPVariable_det(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "det(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::det(Tensor self) -> Tensor
  auto dispatch_det = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.det();
  };
  return wrap(dispatch_det(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// detach
static PyObject * THPVariable_detach(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "detach(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::detach(Tensor self) -> Tensor
  auto dispatch_detach = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.detach();
  };
  return wrap(dispatch_detach(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// detach_
static PyObject * THPVariable_detach_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "detach_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::detach_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_detach_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.detach_();
  };
  return wrap(dispatch_detach_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diag
static PyObject * THPVariable_diag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diag(Tensor input, int64_t diagonal=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::diag(Tensor self, int diagonal=0) -> Tensor
    auto dispatch_diag = [](const Tensor & self, int64_t diagonal) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.diag(diagonal);
    };
    return wrap(dispatch_diag(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::diag.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_diag_out = [](Tensor out, const Tensor & self, int64_t diagonal) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::diag_out(out, self, diagonal);
    };
    return wrap(dispatch_diag_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diag_embed
static PyObject * THPVariable_diag_embed(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diag_embed(Tensor input, int64_t offset=0, int64_t dim1=-2, int64_t dim2=-1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor
  auto dispatch_diag_embed = [](const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.diag_embed(offset, dim1, dim2);
  };
  return wrap(dispatch_diag_embed(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toInt64(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diagflat
static PyObject * THPVariable_diagflat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diagflat(Tensor input, int64_t offset=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::diagflat(Tensor self, int offset=0) -> Tensor
  auto dispatch_diagflat = [](const Tensor & self, int64_t offset) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.diagflat(offset);
  };
  return wrap(dispatch_diagflat(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// diagonal
static PyObject * THPVariable_diagonal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diagonal(Tensor input, *, Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset=0)",
    "diagonal(Tensor input, int64_t offset=0, int64_t dim1=0, int64_t dim2=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::diagonal.Dimname(Tensor(a) self, *, Dimname outdim, Dimname dim1, Dimname dim2, int offset=0) -> Tensor(a)
      auto dispatch_diagonal = [](const Tensor & self, Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.diagonal(outdim, dim1, dim2, offset);
      };
      return wrap(dispatch_diagonal(_r.tensor(0), _r.dimname(1), _r.dimname(2), _r.dimname(3), _r.toInt64(4)));
    }
    case 1: {
      // aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
      auto dispatch_diagonal = [](const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.diagonal(offset, dim1, dim2);
      };
      return wrap(dispatch_diagonal(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toInt64(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// digamma
static PyObject * THPVariable_digamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "digamma(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::digamma(Tensor self) -> Tensor
    auto dispatch_digamma = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.digamma();
    };
    return wrap(dispatch_digamma(_r.tensor(0)));
  } else {
    // aten::digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_digamma_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::digamma_out(out, self);
    };
    return wrap(dispatch_digamma_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// dist
static PyObject * THPVariable_dist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dist(Tensor input, Tensor other, Scalar p=2)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor
  auto dispatch_dist = [](const Tensor & self, const Tensor & other, Scalar p) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.dist(other, p);
  };
  return wrap(dispatch_dist(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// div
static PyObject * THPVariable_div(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "div(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::div.Tensor(Tensor self, Tensor other) -> Tensor
    auto dispatch_div = [](const Tensor & self, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.div(other);
    };
    return wrap(dispatch_div(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_div_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::div_out(out, self, other);
    };
    return wrap(dispatch_div_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// dot
static PyObject * THPVariable_dot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dot(Tensor input, Tensor tensor, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::dot(Tensor self, Tensor tensor) -> Tensor
    auto dispatch_dot = [](const Tensor & self, const Tensor & tensor) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.dot(tensor);
    };
    return wrap(dispatch_dot(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_dot_out = [](Tensor out, const Tensor & self, const Tensor & tensor) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::dot_out(out, self, tensor);
    };
    return wrap(dispatch_dot_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// dropout
static PyObject * THPVariable_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dropout(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::dropout(Tensor input, float p, bool train) -> Tensor
  auto dispatch_dropout = [](const Tensor & input, double p, bool train) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::dropout(input, p, train);
  };
  return wrap(dispatch_dropout(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// dropout_
static PyObject * THPVariable_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dropout_(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
  auto dispatch_dropout_ = [](Tensor self, double p, bool train) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::dropout_(self, p, train);
  };
  return wrap(dispatch_dropout_(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
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
    static PyStructSequence_Desc desc = { "torch.return_types.eig_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.eig", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "eig(Tensor input, bool eigenvectors=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::eig(Tensor self, bool eigenvectors=False) -> (Tensor eigenvalues, Tensor eigenvectors)
    auto dispatch_eig = [](const Tensor & self, bool eigenvectors) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.eig(eigenvectors);
    };
    return wrap(&NamedTuple1, dispatch_eig(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::eig.e(Tensor self, bool eigenvectors=False, *, Tensor(a!) e, Tensor(b!) v) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_eig_out = [](Tensor & e, Tensor & v, const Tensor & self, bool eigenvectors) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::eig_out(e, v, self, eigenvectors);
    };
    return wrap(&NamedTuple, dispatch_eig_out(out[0], out[1], _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// einsum
static PyObject * THPVariable_einsum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "einsum(std::string equation, TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::einsum(str equation, Tensor[] tensors) -> Tensor
  auto dispatch_einsum = [](std::string equation, TensorList tensors) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::einsum(equation, tensors);
  };
  return wrap(dispatch_einsum(_r.string(0), _r.tensorlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// embedding
static PyObject * THPVariable_embedding(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "embedding(Tensor weight, Tensor indices, int64_t padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
  auto dispatch_embedding = [](const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
  };
  return wrap(dispatch_embedding(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toBool(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// embedding_bag
static PyObject * THPVariable_embedding_bag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int64_t mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)
  auto dispatch_embedding_bag = [](const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights, bool include_last_offset) -> std::tuple<Tensor,Tensor,Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
  };
  return wrap(dispatch_embedding_bag(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3), _r.toInt64(4), _r.toBool(5), _r.tensor(6), _r.toBool(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// embedding_renorm_
static PyObject * THPVariable_embedding_renorm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "embedding_renorm_(Tensor input, Tensor indices, double max_norm, double norm_type)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)
  auto dispatch_embedding_renorm_ = [](Tensor self, const Tensor & indices, double max_norm, double norm_type) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::embedding_renorm_(self, indices, max_norm, norm_type);
  };
  return wrap(dispatch_embedding_renorm_(_r.tensor(0), _r.tensor(1), _r.toDouble(2), _r.toDouble(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// empty
static PyObject * THPVariable_empty(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "empty(IntArrayRef size, *, DimnameList? names, MemoryFormat? memory_format=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "empty(IntArrayRef size, *, MemoryFormat? memory_format=None, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::empty.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
      auto __names = _r.toDimnameListOptional(1);
      c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartype(3))
          .device(_r.device(5))
          .layout(_r.layout(4).layout)
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_empty = [](IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::empty(size, names, options, memory_format);
      };
      return wrap(dispatch_empty(_r.intlist(0), names, options, _r.memoryformatOptional(2)));
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartype(3))
            .device(_r.device(5))
            .layout(_r.layout(4).layout)
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch::utils::maybe_initialize_cuda(options);
        auto dispatch_empty = [](IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::empty(size, options, memory_format);
        };
        return wrap(dispatch_empty(_r.intlist(0), options, _r.memoryformatOptional(1)));
      } else {
        // aten::empty.out(int[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartype(3), _r.isNone(3),
                               _r.layout(4), _r.isNone(4),
                               _r.device(5), _r.isNone(5));
        auto dispatch_empty_out = [](Tensor out, IntArrayRef size, c10::optional<MemoryFormat> memory_format) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::empty_out(out, size, memory_format);
        };
        return wrap(dispatch_empty_out(_r.tensor(2), _r.intlist(0), _r.memoryformatOptional(1)).set_requires_grad(_r.toBool(7)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// empty_like
static PyObject * THPVariable_empty_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "empty_like(Tensor input, *, MemoryFormat? memory_format=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "empty_like(Tensor input, *, MemoryFormat? memory_format=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::empty_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor
      auto self = _r.tensor(0);
      const auto options = TensorOptions()
          .dtype(_r.scalartypeWithDefault(2, self.scalar_type()))
          .device(_r.deviceWithDefault(4, self.device()))
          .layout(_r.layoutWithDefault(3, *torch::getLayout(self.options().backend())).layout)
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_empty_like = [](const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::empty_like(self, options, memory_format);
      };
      return wrap(dispatch_empty_like(self, options, _r.memoryformatOptional(1)));
    }
    case 1: {
      // aten::empty_like(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
      auto dispatch_empty_like = [](const Tensor & self, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::empty_like(self, memory_format);
      };
      return wrap(dispatch_empty_like(_r.tensor(0), _r.memoryformatOptional(1)).set_requires_grad(_r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// empty_strided
static PyObject * THPVariable_empty_strided(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "empty_strided(IntArrayRef size, IntArrayRef stride, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartype(2))
      .device(_r.device(4))
      .layout(_r.layout(3).layout)
      .requires_grad(_r.toBool(6))
      .pinned_memory(_r.toBool(5));
  torch::utils::maybe_initialize_cuda(options);
  auto dispatch_empty_strided = [](IntArrayRef size, IntArrayRef stride, const TensorOptions & options) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::empty_strided(size, stride, options);
  };
  return wrap(dispatch_empty_strided(_r.intlist(0), _r.intlist(1), options));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// eq
static PyObject * THPVariable_eq(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eq(Tensor input, Tensor other, *, Tensor out=None)",
    "eq(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::eq.Tensor(Tensor self, Tensor other) -> Tensor
        auto dispatch_eq = [](const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.eq(other);
        };
        return wrap(dispatch_eq(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_eq_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::eq_out(out, self, other);
        };
        return wrap(dispatch_eq_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::eq.Scalar(Tensor self, Scalar other) -> Tensor
        auto dispatch_eq = [](const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.eq(other);
        };
        return wrap(dispatch_eq(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_eq_out = [](Tensor out, const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::eq_out(out, self, other);
        };
        return wrap(dispatch_eq_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// equal
static PyObject * THPVariable_equal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "equal(Tensor input, Tensor other)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::equal(Tensor self, Tensor other) -> bool
  auto dispatch_equal = [](const Tensor & self, const Tensor & other) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.equal(other);
  };
  return wrap(dispatch_equal(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// erf
static PyObject * THPVariable_erf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erf(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::erf(Tensor self) -> Tensor
    auto dispatch_erf = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.erf();
    };
    return wrap(dispatch_erf(_r.tensor(0)));
  } else {
    // aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_erf_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::erf_out(out, self);
    };
    return wrap(dispatch_erf_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// erf_
static PyObject * THPVariable_erf_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erf_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::erf_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_erf_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erf_();
  };
  return wrap(dispatch_erf_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// erfc
static PyObject * THPVariable_erfc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erfc(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::erfc(Tensor self) -> Tensor
    auto dispatch_erfc = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.erfc();
    };
    return wrap(dispatch_erfc(_r.tensor(0)));
  } else {
    // aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_erfc_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::erfc_out(out, self);
    };
    return wrap(dispatch_erfc_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// erfc_
static PyObject * THPVariable_erfc_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erfc_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::erfc_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_erfc_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erfc_();
  };
  return wrap(dispatch_erfc_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// erfinv
static PyObject * THPVariable_erfinv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erfinv(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::erfinv(Tensor self) -> Tensor
    auto dispatch_erfinv = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.erfinv();
    };
    return wrap(dispatch_erfinv(_r.tensor(0)));
  } else {
    // aten::erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_erfinv_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::erfinv_out(out, self);
    };
    return wrap(dispatch_erfinv_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// exp
static PyObject * THPVariable_exp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "exp(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::exp(Tensor self) -> Tensor
    auto dispatch_exp = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.exp();
    };
    return wrap(dispatch_exp(_r.tensor(0)));
  } else {
    // aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_exp_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::exp_out(out, self);
    };
    return wrap(dispatch_exp_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// exp_
static PyObject * THPVariable_exp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "exp_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::exp_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_exp_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.exp_();
  };
  return wrap(dispatch_exp_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// expm1
static PyObject * THPVariable_expm1(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "expm1(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::expm1(Tensor self) -> Tensor
    auto dispatch_expm1 = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.expm1();
    };
    return wrap(dispatch_expm1(_r.tensor(0)));
  } else {
    // aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_expm1_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::expm1_out(out, self);
    };
    return wrap(dispatch_expm1_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// expm1_
static PyObject * THPVariable_expm1_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "expm1_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::expm1_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_expm1_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.expm1_();
  };
  return wrap(dispatch_expm1_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// eye
static PyObject * THPVariable_eye(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eye(int64_t n, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "eye(int64_t n, int64_t m, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(1)) {
        // aten::eye(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartype(2))
            .device(_r.device(4))
            .layout(_r.layout(3).layout)
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch::utils::maybe_initialize_cuda(options);
        auto dispatch_eye = [](int64_t n, const TensorOptions & options) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::eye(n, options);
        };
        return wrap(dispatch_eye(_r.toInt64(0), options));
      } else {
        // aten::eye.out(int n, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartype(2), _r.isNone(2),
                               _r.layout(3), _r.isNone(3),
                               _r.device(4), _r.isNone(4));
        auto dispatch_eye_out = [](Tensor out, int64_t n) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::eye_out(out, n);
        };
        return wrap(dispatch_eye_out(_r.tensor(1), _r.toInt64(0)).set_requires_grad(_r.toBool(6)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::eye.m(int n, int m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartype(3))
            .device(_r.device(5))
            .layout(_r.layout(4).layout)
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch::utils::maybe_initialize_cuda(options);
        auto dispatch_eye = [](int64_t n, int64_t m, const TensorOptions & options) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::eye(n, m, options);
        };
        return wrap(dispatch_eye(_r.toInt64(0), _r.toInt64(1), options));
      } else {
        // aten::eye.m_out(int n, int m, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartype(3), _r.isNone(3),
                               _r.layout(4), _r.isNone(4),
                               _r.device(5), _r.isNone(5));
        auto dispatch_eye_out = [](Tensor out, int64_t n, int64_t m) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::eye_out(out, n, m);
        };
        return wrap(dispatch_eye_out(_r.tensor(2), _r.toInt64(0), _r.toInt64(1)).set_requires_grad(_r.toBool(7)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fake_quantize_per_channel_affine
static PyObject * THPVariable_fake_quantize_per_channel_affine(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fake_quantize_per_channel_affine(Tensor input, Tensor scale, Tensor zero_point, int64_t axis, int64_t quant_min, int64_t quant_max)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::fake_quantize_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> Tensor
  auto dispatch_fake_quantize_per_channel_affine = [](const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fake_quantize_per_channel_affine(self, scale, zero_point, axis, quant_min, quant_max);
  };
  return wrap(dispatch_fake_quantize_per_channel_affine(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4), _r.toInt64(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fake_quantize_per_tensor_affine
static PyObject * THPVariable_fake_quantize_per_tensor_affine(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fake_quantize_per_tensor_affine(Tensor input, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor
  auto dispatch_fake_quantize_per_tensor_affine = [](const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fake_quantize_per_tensor_affine(self, scale, zero_point, quant_min, quant_max);
  };
  return wrap(dispatch_fake_quantize_per_tensor_affine(_r.tensor(0), _r.toDouble(1), _r.toInt64(2), _r.toInt64(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fbgemm_linear_fp16_weight
static PyObject * THPVariable_fbgemm_linear_fp16_weight(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fbgemm_linear_fp16_weight(Tensor input, Tensor packed_weight, Tensor bias)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::fbgemm_linear_fp16_weight(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor
  auto dispatch_fbgemm_linear_fp16_weight = [](const Tensor & input, const Tensor & packed_weight, const Tensor & bias) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fbgemm_linear_fp16_weight(input, packed_weight, bias);
  };
  return wrap(dispatch_fbgemm_linear_fp16_weight(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fbgemm_linear_fp16_weight_fp32_activation
static PyObject * THPVariable_fbgemm_linear_fp16_weight_fp32_activation(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fbgemm_linear_fp16_weight_fp32_activation(Tensor input, Tensor packed_weight, Tensor bias)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::fbgemm_linear_fp16_weight_fp32_activation(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor
  auto dispatch_fbgemm_linear_fp16_weight_fp32_activation = [](const Tensor & input, const Tensor & packed_weight, const Tensor & bias) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fbgemm_linear_fp16_weight_fp32_activation(input, packed_weight, bias);
  };
  return wrap(dispatch_fbgemm_linear_fp16_weight_fp32_activation(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fbgemm_linear_int8_weight
static PyObject * THPVariable_fbgemm_linear_int8_weight(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor
  auto dispatch_fbgemm_linear_int8_weight = [](const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, Scalar weight_scale, Scalar weight_zero_point, const Tensor & bias) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fbgemm_linear_int8_weight(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  };
  return wrap(dispatch_fbgemm_linear_int8_weight(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.scalar(4), _r.scalar(5), _r.tensor(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fbgemm_linear_int8_weight_fp32_activation
static PyObject * THPVariable_fbgemm_linear_int8_weight_fp32_activation(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fbgemm_linear_int8_weight_fp32_activation(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::fbgemm_linear_int8_weight_fp32_activation(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor
  auto dispatch_fbgemm_linear_int8_weight_fp32_activation = [](const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, Scalar weight_scale, Scalar weight_zero_point, const Tensor & bias) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fbgemm_linear_int8_weight_fp32_activation(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  };
  return wrap(dispatch_fbgemm_linear_int8_weight_fp32_activation(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.scalar(4), _r.scalar(5), _r.tensor(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fbgemm_linear_quantize_weight
static PyObject * THPVariable_fbgemm_linear_quantize_weight(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fbgemm_linear_quantize_weight(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::fbgemm_linear_quantize_weight(Tensor input) -> (Tensor, Tensor, float, int)
  auto dispatch_fbgemm_linear_quantize_weight = [](const Tensor & input) -> std::tuple<Tensor,Tensor,double,int64_t> {
    pybind11::gil_scoped_release no_gil;
    return at::fbgemm_linear_quantize_weight(input);
  };
  return wrap(dispatch_fbgemm_linear_quantize_weight(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fbgemm_pack_gemm_matrix_fp16
static PyObject * THPVariable_fbgemm_pack_gemm_matrix_fp16(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fbgemm_pack_gemm_matrix_fp16(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::fbgemm_pack_gemm_matrix_fp16(Tensor input) -> Tensor
  auto dispatch_fbgemm_pack_gemm_matrix_fp16 = [](const Tensor & input) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fbgemm_pack_gemm_matrix_fp16(input);
  };
  return wrap(dispatch_fbgemm_pack_gemm_matrix_fp16(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// fbgemm_pack_quantized_matrix
static PyObject * THPVariable_fbgemm_pack_quantized_matrix(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fbgemm_pack_quantized_matrix(Tensor input)",
    "fbgemm_pack_quantized_matrix(Tensor input, int64_t K, int64_t N)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::fbgemm_pack_quantized_matrix(Tensor input) -> Tensor
      auto dispatch_fbgemm_pack_quantized_matrix = [](const Tensor & input) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::fbgemm_pack_quantized_matrix(input);
      };
      return wrap(dispatch_fbgemm_pack_quantized_matrix(_r.tensor(0)));
    }
    case 1: {
      // aten::fbgemm_pack_quantized_matrix.KN(Tensor input, int K, int N) -> Tensor
      auto dispatch_fbgemm_pack_quantized_matrix = [](const Tensor & input, int64_t K, int64_t N) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::fbgemm_pack_quantized_matrix(input, K, N);
      };
      return wrap(dispatch_fbgemm_pack_quantized_matrix(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// feature_alpha_dropout
static PyObject * THPVariable_feature_alpha_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "feature_alpha_dropout(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor
  auto dispatch_feature_alpha_dropout = [](const Tensor & input, double p, bool train) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::feature_alpha_dropout(input, p, train);
  };
  return wrap(dispatch_feature_alpha_dropout(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// feature_alpha_dropout_
static PyObject * THPVariable_feature_alpha_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "feature_alpha_dropout_(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::feature_alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
  auto dispatch_feature_alpha_dropout_ = [](Tensor self, double p, bool train) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::feature_alpha_dropout_(self, p, train);
  };
  return wrap(dispatch_feature_alpha_dropout_(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// feature_dropout
static PyObject * THPVariable_feature_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "feature_dropout(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::feature_dropout(Tensor input, float p, bool train) -> Tensor
  auto dispatch_feature_dropout = [](const Tensor & input, double p, bool train) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::feature_dropout(input, p, train);
  };
  return wrap(dispatch_feature_dropout(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// feature_dropout_
static PyObject * THPVariable_feature_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "feature_dropout_(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::feature_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
  auto dispatch_feature_dropout_ = [](Tensor self, double p, bool train) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::feature_dropout_(self, p, train);
  };
  return wrap(dispatch_feature_dropout_(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft
static PyObject * THPVariable_fft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft(Tensor input, int64_t signal_ndim, bool normalized=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::fft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor
  auto dispatch_fft = [](const Tensor & self, int64_t signal_ndim, bool normalized) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.fft(signal_ndim, normalized);
  };
  return wrap(dispatch_fft(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// fill_
static PyObject * THPVariable_fill_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fill_(Tensor input, Tensor value)",
    "fill_(Tensor input, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
      auto dispatch_fill_ = [](Tensor self, const Tensor & value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fill_(value);
      };
      return wrap(dispatch_fill_(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
      auto dispatch_fill_ = [](Tensor self, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fill_(value);
      };
      return wrap(dispatch_fill_(_r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// flatten
static PyObject * THPVariable_flatten(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "flatten(Tensor input, Dimname start_dim, Dimname end_dim, Dimname out_dim)",
    "flatten(Tensor input, DimnameList dims, Dimname out_dim)",
    "flatten(Tensor input, int64_t start_dim, int64_t end_dim, Dimname out_dim)",
    "flatten(Tensor input, int64_t start_dim=0, int64_t end_dim=-1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::flatten.using_names(Tensor self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor
      auto dispatch_flatten = [](const Tensor & self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(start_dim, end_dim, out_dim);
      };
      return wrap(dispatch_flatten(_r.tensor(0), _r.dimname(1), _r.dimname(2), _r.dimname(3)));
    }
    case 1: {
      // aten::flatten.DimnameList(Tensor self, Dimname[] dims, Dimname out_dim) -> Tensor
      auto dispatch_flatten = [](const Tensor & self, DimnameList dims, Dimname out_dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(dims, out_dim);
      };
      return wrap(dispatch_flatten(_r.tensor(0), _r.dimnamelist(1), _r.dimname(2)));
    }
    case 2: {
      // aten::flatten.named_out_dim(Tensor self, int start_dim, int end_dim, Dimname out_dim) -> Tensor
      auto dispatch_flatten = [](const Tensor & self, int64_t start_dim, int64_t end_dim, Dimname out_dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(start_dim, end_dim, out_dim);
      };
      return wrap(dispatch_flatten(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.dimname(3)));
    }
    case 3: {
      // aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor
      auto dispatch_flatten = [](const Tensor & self, int64_t start_dim, int64_t end_dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(start_dim, end_dim);
      };
      return wrap(dispatch_flatten(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// flip
static PyObject * THPVariable_flip(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "flip(Tensor input, IntArrayRef dims)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::flip(Tensor self, int[] dims) -> Tensor
  auto dispatch_flip = [](const Tensor & self, IntArrayRef dims) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.flip(dims);
  };
  return wrap(dispatch_flip(_r.tensor(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// floor
static PyObject * THPVariable_floor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "floor(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::floor(Tensor self) -> Tensor
    auto dispatch_floor = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.floor();
    };
    return wrap(dispatch_floor(_r.tensor(0)));
  } else {
    // aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_floor_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::floor_out(out, self);
    };
    return wrap(dispatch_floor_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// floor_
static PyObject * THPVariable_floor_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "floor_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::floor_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_floor_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.floor_();
  };
  return wrap(dispatch_floor_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// floor_divide
static PyObject * THPVariable_floor_divide(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "floor_divide(Tensor input, Tensor other)",
    "floor_divide(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::floor_divide(Tensor input, Tensor other) -> Tensor
      auto dispatch_floor_divide = [](const Tensor & input, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::floor_divide(input, other);
      };
      return wrap(dispatch_floor_divide(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::floor_divide.Scalar(Tensor input, Scalar other) -> Tensor
      auto dispatch_floor_divide = [](const Tensor & input, Scalar other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::floor_divide(input, other);
      };
      return wrap(dispatch_floor_divide(_r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// fmod
static PyObject * THPVariable_fmod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fmod(Tensor input, Tensor other, *, Tensor out=None)",
    "fmod(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor
        auto dispatch_fmod = [](const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.fmod(other);
        };
        return wrap(dispatch_fmod(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::fmod.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_fmod_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::fmod_out(out, self, other);
        };
        return wrap(dispatch_fmod_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor
        auto dispatch_fmod = [](const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.fmod(other);
        };
        return wrap(dispatch_fmod(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::fmod.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_fmod_out = [](Tensor out, const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::fmod_out(out, self, other);
        };
        return wrap(dispatch_fmod_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// frac
static PyObject * THPVariable_frac(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "frac(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::frac(Tensor self) -> Tensor
    auto dispatch_frac = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.frac();
    };
    return wrap(dispatch_frac(_r.tensor(0)));
  } else {
    // aten::frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_frac_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::frac_out(out, self);
    };
    return wrap(dispatch_frac_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// frac_
static PyObject * THPVariable_frac_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "frac_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::frac_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_frac_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.frac_();
  };
  return wrap(dispatch_frac_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// frobenius_norm
static PyObject * THPVariable_frobenius_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "frobenius_norm(Tensor input)",
    "frobenius_norm(Tensor input, IntArrayRef[1] dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::frobenius_norm(Tensor self) -> Tensor
      auto dispatch_frobenius_norm = [](const Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::frobenius_norm(self);
      };
      return wrap(dispatch_frobenius_norm(_r.tensor(0)));
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
        auto dispatch_frobenius_norm = [](const Tensor & self, IntArrayRef dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::frobenius_norm(self, dim, keepdim);
        };
        return wrap(dispatch_frobenius_norm(_r.tensor(0), _r.intlist(1), _r.toBool(2)));
      } else {
        // aten::frobenius_norm.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_frobenius_norm_out = [](Tensor out, const Tensor & self, IntArrayRef dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::frobenius_norm_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_frobenius_norm_out(_r.tensor(3), _r.tensor(0), _r.intlist(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// from_file
static PyObject * THPVariable_from_file(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "from_file(std::string filename, bool? shared=None, int64_t? size=0, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartype(3))
      .device(_r.device(5))
      .layout(_r.layout(4).layout)
      .requires_grad(_r.toBool(7))
      .pinned_memory(_r.toBool(6));
  torch::utils::maybe_initialize_cuda(options);
  auto dispatch_from_file = [](std::string filename, c10::optional<bool> shared, c10::optional<int64_t> size, const TensorOptions & options) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::from_file(filename, shared, size, options);
  };
  return wrap(dispatch_from_file(_r.string(0), _r.toBoolOptional(1), _r.toInt64Optional(2), options));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// full
static PyObject * THPVariable_full(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "full(IntArrayRef size, Scalar fill_value, *, DimnameList? names, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "full(IntArrayRef size, Scalar fill_value, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::full.names(int[] size, Scalar fill_value, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(2);
      c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartype(3))
          .device(_r.device(5))
          .layout(_r.layout(4).layout)
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_full = [](IntArrayRef size, Scalar fill_value, c10::optional<DimnameList> names, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::full(size, fill_value, names, options);
      };
      return wrap(dispatch_full(_r.intlist(0), _r.scalar(1), names, options));
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartype(3))
            .device(_r.device(5))
            .layout(_r.layout(4).layout)
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch::utils::maybe_initialize_cuda(options);
        auto dispatch_full = [](IntArrayRef size, Scalar fill_value, const TensorOptions & options) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::full(size, fill_value, options);
        };
        return wrap(dispatch_full(_r.intlist(0), _r.scalar(1), options));
      } else {
        // aten::full.out(int[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartype(3), _r.isNone(3),
                               _r.layout(4), _r.isNone(4),
                               _r.device(5), _r.isNone(5));
        auto dispatch_full_out = [](Tensor out, IntArrayRef size, Scalar fill_value) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::full_out(out, size, fill_value);
        };
        return wrap(dispatch_full_out(_r.tensor(2), _r.intlist(0), _r.scalar(1)).set_requires_grad(_r.toBool(7)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// full_like
static PyObject * THPVariable_full_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "full_like(Tensor input, Scalar fill_value, *, MemoryFormat? memory_format=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "full_like(Tensor input, Scalar fill_value, *, MemoryFormat? memory_format=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::full_like.dtype(Tensor self, Scalar fill_value, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor
      auto self = _r.tensor(0);
      const auto options = TensorOptions()
          .dtype(_r.scalartypeWithDefault(3, self.scalar_type()))
          .device(_r.deviceWithDefault(5, self.device()))
          .layout(_r.layoutWithDefault(4, *torch::getLayout(self.options().backend())).layout)
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_full_like = [](const Tensor & self, Scalar fill_value, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::full_like(self, fill_value, options, memory_format);
      };
      return wrap(dispatch_full_like(self, _r.scalar(1), options, _r.memoryformatOptional(2)));
    }
    case 1: {
      // aten::full_like(Tensor self, Scalar fill_value, *, MemoryFormat? memory_format=None) -> Tensor
      auto dispatch_full_like = [](const Tensor & self, Scalar fill_value, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::full_like(self, fill_value, memory_format);
      };
      return wrap(dispatch_full_like(_r.tensor(0), _r.scalar(1), _r.memoryformatOptional(2)).set_requires_grad(_r.toBool(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// gather
static PyObject * THPVariable_gather(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gather(Tensor input, Dimname dim, Tensor index, *, bool sparse_grad=False, Tensor out=None)",
    "gather(Tensor input, int64_t dim, Tensor index, *, bool sparse_grad=False, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::gather.dimname(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False) -> Tensor
        auto dispatch_gather = [](const Tensor & self, Dimname dim, const Tensor & index, bool sparse_grad) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.gather(dim, index, sparse_grad);
        };
        return wrap(dispatch_gather(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.toBool(3)));
      } else {
        // aten::gather.dimname_out(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_gather_out = [](Tensor out, const Tensor & self, Dimname dim, const Tensor & index, bool sparse_grad) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::gather_out(out, self, dim, index, sparse_grad);
        };
        return wrap(dispatch_gather_out(_r.tensor(4), _r.tensor(0), _r.dimname(1), _r.tensor(2), _r.toBool(3)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
        auto dispatch_gather = [](const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.gather(dim, index, sparse_grad);
        };
        return wrap(dispatch_gather(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.toBool(3)));
      } else {
        // aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_gather_out = [](Tensor out, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::gather_out(out, self, dim, index, sparse_grad);
        };
        return wrap(dispatch_gather_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.toBool(3)));
      }
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
  static PythonArgParser parser({
    "ge(Tensor input, Tensor other, *, Tensor out=None)",
    "ge(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::ge.Tensor(Tensor self, Tensor other) -> Tensor
        auto dispatch_ge = [](const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.ge(other);
        };
        return wrap(dispatch_ge(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_ge_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::ge_out(out, self, other);
        };
        return wrap(dispatch_ge_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::ge.Scalar(Tensor self, Scalar other) -> Tensor
        auto dispatch_ge = [](const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.ge(other);
        };
        return wrap(dispatch_ge(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_ge_out = [](Tensor out, const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::ge_out(out, self, other);
        };
        return wrap(dispatch_ge_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// geqrf
static PyObject * THPVariable_geqrf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"a", ""}, {"tau", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.geqrf_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.geqrf", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "geqrf(Tensor input, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)
    auto dispatch_geqrf = [](const Tensor & self) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.geqrf();
    };
    return wrap(&NamedTuple1, dispatch_geqrf(_r.tensor(0)));
  } else {
    // aten::geqrf.a(Tensor self, *, Tensor(a!) a, Tensor(b!) tau) -> (Tensor(a!) a, Tensor(b!) tau)
    auto out = _r.tensorlist_n<2>(1);
    auto dispatch_geqrf_out = [](Tensor & a, Tensor & tau, const Tensor & self) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::geqrf_out(a, tau, self);
    };
    return wrap(&NamedTuple, dispatch_geqrf_out(out[0], out[1], _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ger
static PyObject * THPVariable_ger(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ger(Tensor input, Tensor vec2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::ger(Tensor self, Tensor vec2) -> Tensor
    auto dispatch_ger = [](const Tensor & self, const Tensor & vec2) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.ger(vec2);
    };
    return wrap(dispatch_ger(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::ger.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_ger_out = [](Tensor out, const Tensor & self, const Tensor & vec2) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::ger_out(out, self, vec2);
    };
    return wrap(dispatch_ger_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// grid_sampler
static PyObject * THPVariable_grid_sampler(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "grid_sampler(Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
  auto dispatch_grid_sampler = [](const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners);
  };
  return wrap(dispatch_grid_sampler(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toInt64(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// grid_sampler_2d
static PyObject * THPVariable_grid_sampler_2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "grid_sampler_2d(Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
  auto dispatch_grid_sampler_2d = [](const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners);
  };
  return wrap(dispatch_grid_sampler_2d(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toInt64(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// grid_sampler_3d
static PyObject * THPVariable_grid_sampler_3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "grid_sampler_3d(Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
  auto dispatch_grid_sampler_3d = [](const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners);
  };
  return wrap(dispatch_grid_sampler_3d(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toInt64(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// group_norm
static PyObject * THPVariable_group_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "group_norm(Tensor input, int64_t num_groups, Tensor? weight=None, Tensor? bias=None, double eps=1e-05, bool cudnn_enabled=True)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
  auto dispatch_group_norm = [](const Tensor & input, int64_t num_groups, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enabled) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::group_norm(input, num_groups, weight, bias, eps, cudnn_enabled);
  };
  return wrap(dispatch_group_norm(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3), _r.toDouble(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// gru
static PyObject * THPVariable_gru(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gru(Tensor data, Tensor batch_sizes, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)",
    "gru(Tensor input, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::gru.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
      auto dispatch_gru = [](const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::gru(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
      };
      return wrap(dispatch_gru(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensorlist(3), _r.toBool(4), _r.toInt64(5), _r.toDouble(6), _r.toBool(7), _r.toBool(8)));
    }
    case 1: {
      // aten::gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
      auto dispatch_gru = [](const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
      };
      return wrap(dispatch_gru(_r.tensor(0), _r.tensor(1), _r.tensorlist(2), _r.toBool(3), _r.toInt64(4), _r.toDouble(5), _r.toBool(6), _r.toBool(7), _r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// gru_cell
static PyObject * THPVariable_gru_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
  auto dispatch_gru_cell = [](const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  };
  return wrap(dispatch_gru_cell(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// gt
static PyObject * THPVariable_gt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gt(Tensor input, Tensor other, *, Tensor out=None)",
    "gt(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::gt.Tensor(Tensor self, Tensor other) -> Tensor
        auto dispatch_gt = [](const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.gt(other);
        };
        return wrap(dispatch_gt(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_gt_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::gt_out(out, self, other);
        };
        return wrap(dispatch_gt_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::gt.Scalar(Tensor self, Scalar other) -> Tensor
        auto dispatch_gt = [](const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.gt(other);
        };
        return wrap(dispatch_gt(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_gt_out = [](Tensor out, const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::gt_out(out, self, other);
        };
        return wrap(dispatch_gt_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// hamming_window
static PyObject * THPVariable_hamming_window(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hamming_window(int64_t window_length, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "hamming_window(int64_t window_length, bool periodic, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "hamming_window(int64_t window_length, bool periodic, double alpha, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "hamming_window(int64_t window_length, bool periodic, double alpha, double beta, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::hamming_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartype(1))
          .device(_r.device(3))
          .layout(_r.layout(2).layout)
          .requires_grad(_r.toBool(5))
          .pinned_memory(_r.toBool(4));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_hamming_window = [](int64_t window_length, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::hamming_window(window_length, options);
      };
      return wrap(dispatch_hamming_window(_r.toInt64(0), options));
    }
    case 1: {
      // aten::hamming_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartype(2))
          .device(_r.device(4))
          .layout(_r.layout(3).layout)
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_hamming_window = [](int64_t window_length, bool periodic, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::hamming_window(window_length, periodic, options);
      };
      return wrap(dispatch_hamming_window(_r.toInt64(0), _r.toBool(1), options));
    }
    case 2: {
      // aten::hamming_window.periodic_alpha(int window_length, bool periodic, float alpha, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartype(3))
          .device(_r.device(5))
          .layout(_r.layout(4).layout)
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_hamming_window = [](int64_t window_length, bool periodic, double alpha, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::hamming_window(window_length, periodic, alpha, options);
      };
      return wrap(dispatch_hamming_window(_r.toInt64(0), _r.toBool(1), _r.toDouble(2), options));
    }
    case 3: {
      // aten::hamming_window.periodic_alpha_beta(int window_length, bool periodic, float alpha, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartype(4))
          .device(_r.device(6))
          .layout(_r.layout(5).layout)
          .requires_grad(_r.toBool(8))
          .pinned_memory(_r.toBool(7));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_hamming_window = [](int64_t window_length, bool periodic, double alpha, double beta, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::hamming_window(window_length, periodic, alpha, beta, options);
      };
      return wrap(dispatch_hamming_window(_r.toInt64(0), _r.toBool(1), _r.toDouble(2), _r.toDouble(3), options));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// hann_window
static PyObject * THPVariable_hann_window(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hann_window(int64_t window_length, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "hann_window(int64_t window_length, bool periodic, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartype(1))
          .device(_r.device(3))
          .layout(_r.layout(2).layout)
          .requires_grad(_r.toBool(5))
          .pinned_memory(_r.toBool(4));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_hann_window = [](int64_t window_length, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::hann_window(window_length, options);
      };
      return wrap(dispatch_hann_window(_r.toInt64(0), options));
    }
    case 1: {
      // aten::hann_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartype(2))
          .device(_r.device(4))
          .layout(_r.layout(3).layout)
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_hann_window = [](int64_t window_length, bool periodic, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::hann_window(window_length, periodic, options);
      };
      return wrap(dispatch_hann_window(_r.toInt64(0), _r.toBool(1), options));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hardshrink
static PyObject * THPVariable_hardshrink(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardshrink(Tensor input, Scalar lambd=0.5)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor
  auto dispatch_hardshrink = [](const Tensor & self, Scalar lambd) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.hardshrink(lambd);
  };
  return wrap(dispatch_hardshrink(_r.tensor(0), _r.scalar(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hinge_embedding_loss
static PyObject * THPVariable_hinge_embedding_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hinge_embedding_loss(Tensor input, Tensor target, double margin=1.0, int64_t reduction=at::Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::hinge_embedding_loss(Tensor self, Tensor target, float margin=1.0, int reduction=Mean) -> Tensor
  auto dispatch_hinge_embedding_loss = [](const Tensor & self, const Tensor & target, double margin, int64_t reduction) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::hinge_embedding_loss(self, target, margin, reduction);
  };
  return wrap(dispatch_hinge_embedding_loss(_r.tensor(0), _r.tensor(1), _r.toDouble(2), _r.toInt64(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// histc
static PyObject * THPVariable_histc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "histc(Tensor input, int64_t bins=100, Scalar min=0, Scalar max=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(4)) {
    // aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
    auto dispatch_histc = [](const Tensor & self, int64_t bins, Scalar min, Scalar max) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.histc(bins, min, max);
    };
    return wrap(dispatch_histc(_r.tensor(0), _r.toInt64(1), _r.scalar(2), _r.scalar(3)));
  } else {
    // aten::histc.out(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_histc_out = [](Tensor out, const Tensor & self, int64_t bins, Scalar min, Scalar max) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::histc_out(out, self, bins, min, max);
    };
    return wrap(dispatch_histc_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.scalar(2), _r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hspmm
static PyObject * THPVariable_hspmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hspmm(Tensor mat1, Tensor mat2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::hspmm(Tensor mat1, Tensor mat2) -> Tensor
    auto dispatch_hspmm = [](const Tensor & mat1, const Tensor & mat2) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hspmm(mat1, mat2);
    };
    return wrap(dispatch_hspmm(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::hspmm.out(Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_hspmm_out = [](Tensor out, const Tensor & mat1, const Tensor & mat2) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hspmm_out(out, mat1, mat2);
    };
    return wrap(dispatch_hspmm_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ifft
static PyObject * THPVariable_ifft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ifft(Tensor input, int64_t signal_ndim, bool normalized=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::ifft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor
  auto dispatch_ifft = [](const Tensor & self, int64_t signal_ndim, bool normalized) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ifft(signal_ndim, normalized);
  };
  return wrap(dispatch_ifft(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// imag
static PyObject * THPVariable_imag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "imag(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::imag(Tensor self) -> Tensor
    auto dispatch_imag = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.imag();
    };
    return wrap(dispatch_imag(_r.tensor(0)));
  } else {
    // aten::imag.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_imag_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::imag_out(out, self);
    };
    return wrap(dispatch_imag_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_add
static PyObject * THPVariable_index_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_add(Tensor input, Dimname dim, Tensor index, Tensor source)",
    "index_add(Tensor input, int64_t dim, Tensor index, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::index_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor source) -> Tensor
      auto dispatch_index_add = [](const Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_add(dim, index, source);
      };
      return wrap(dispatch_index_add(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // aten::index_add(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
      auto dispatch_index_add = [](const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_add(dim, index, source);
      };
      return wrap(dispatch_index_add(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_copy
static PyObject * THPVariable_index_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_copy(Tensor input, Dimname dim, Tensor index, Tensor source)",
    "index_copy(Tensor input, int64_t dim, Tensor index, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::index_copy.dimname(Tensor self, Dimname dim, Tensor index, Tensor source) -> Tensor
      auto dispatch_index_copy = [](const Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_copy(dim, index, source);
      };
      return wrap(dispatch_index_copy(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
      auto dispatch_index_copy = [](const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_copy(dim, index, source);
      };
      return wrap(dispatch_index_copy(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3)));
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
  static PythonArgParser parser({
    "index_fill(Tensor input, Dimname dim, Tensor index, Tensor value)",
    "index_fill(Tensor input, int64_t dim, Tensor index, Tensor value)",
    "index_fill(Tensor input, Dimname dim, Tensor index, Scalar value)",
    "index_fill(Tensor input, int64_t dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::index_fill.Dimname_Tensor(Tensor self, Dimname dim, Tensor index, Tensor value) -> Tensor
      auto dispatch_index_fill = [](const Tensor & self, Dimname dim, const Tensor & index, const Tensor & value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // aten::index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor
      auto dispatch_index_fill = [](const Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3)));
    }
    case 2: {
      // aten::index_fill.Dimname_Scalar(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
      auto dispatch_index_fill = [](const Tensor & self, Dimname dim, const Tensor & index, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.scalar(3)));
    }
    case 3: {
      // aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
      auto dispatch_index_fill = [](const Tensor & self, int64_t dim, const Tensor & index, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// index_put
static PyObject * THPVariable_index_put(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_put(Tensor input, TensorList? indices, Tensor values, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
  auto dispatch_index_put = [](const Tensor & self, TensorList indices, const Tensor & values, bool accumulate) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.index_put(indices, values, accumulate);
  };
  return wrap(dispatch_index_put(_r.tensor(0), _r.tensorlist(1), _r.tensor(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// index_put_
static PyObject * THPVariable_index_put_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_put_(Tensor input, TensorList? indices, Tensor values, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)
  auto dispatch_index_put_ = [](Tensor self, TensorList indices, const Tensor & values, bool accumulate) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.index_put_(indices, values, accumulate);
  };
  return wrap(dispatch_index_put_(_r.tensor(0), _r.tensorlist(1), _r.tensor(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_select
static PyObject * THPVariable_index_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_select(Tensor input, Dimname dim, Tensor index, *, Tensor out=None)",
    "index_select(Tensor input, int64_t dim, Tensor index, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::index_select.dimname(Tensor self, Dimname dim, Tensor index) -> Tensor
        auto dispatch_index_select = [](const Tensor & self, Dimname dim, const Tensor & index) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.index_select(dim, index);
        };
        return wrap(dispatch_index_select(_r.tensor(0), _r.dimname(1), _r.tensor(2)));
      } else {
        // aten::index_select.dimname_out(Tensor self, Dimname dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_index_select_out = [](Tensor out, const Tensor & self, Dimname dim, const Tensor & index) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::index_select_out(out, self, dim, index);
        };
        return wrap(dispatch_index_select_out(_r.tensor(3), _r.tensor(0), _r.dimname(1), _r.tensor(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
        auto dispatch_index_select = [](const Tensor & self, int64_t dim, const Tensor & index) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.index_select(dim, index);
        };
        return wrap(dispatch_index_select(_r.tensor(0), _r.toInt64(1), _r.tensor(2)));
      } else {
        // aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_index_select_out = [](Tensor out, const Tensor & self, int64_t dim, const Tensor & index) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::index_select_out(out, self, dim, index);
        };
        return wrap(dispatch_index_select_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.tensor(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// instance_norm
static PyObject * THPVariable_instance_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor
  auto dispatch_instance_norm = [](const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
  };
  return wrap(dispatch_instance_norm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// int_repr
static PyObject * THPVariable_int_repr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "int_repr(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::int_repr(Tensor self) -> Tensor
  auto dispatch_int_repr = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.int_repr();
  };
  return wrap(dispatch_int_repr(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// inverse
static PyObject * THPVariable_inverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "inverse(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::inverse(Tensor self) -> Tensor
    auto dispatch_inverse = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.inverse();
    };
    return wrap(dispatch_inverse(_r.tensor(0)));
  } else {
    // aten::inverse.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_inverse_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::inverse_out(out, self);
    };
    return wrap(dispatch_inverse_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// irfft
static PyObject * THPVariable_irfft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "irfft(Tensor input, int64_t signal_ndim, bool normalized=False, bool onesided=True, IntArrayRef signal_sizes=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::irfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True, int[] signal_sizes=[]) -> Tensor
  auto dispatch_irfft = [](const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.irfft(signal_ndim, normalized, onesided, signal_sizes);
  };
  return wrap(dispatch_irfft(_r.tensor(0), _r.toInt64(1), _r.toBool(2), _r.toBool(3), _r.intlist(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_complex
static PyObject * THPVariable_is_complex(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_complex(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::is_complex(Tensor self) -> bool
  auto dispatch_is_complex = [](const Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_complex();
  };
  return wrap(dispatch_is_complex(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_distributed
static PyObject * THPVariable_is_distributed(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_distributed(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::is_distributed(Tensor self) -> bool
  auto dispatch_is_distributed = [](const Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_distributed();
  };
  return wrap(dispatch_is_distributed(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_floating_point
static PyObject * THPVariable_is_floating_point(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_floating_point(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::is_floating_point(Tensor self) -> bool
  auto dispatch_is_floating_point = [](const Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_floating_point();
  };
  return wrap(dispatch_is_floating_point(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_nonzero
static PyObject * THPVariable_is_nonzero(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_nonzero(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::is_nonzero(Tensor self) -> bool
  auto dispatch_is_nonzero = [](const Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_nonzero();
  };
  return wrap(dispatch_is_nonzero(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_same_size
static PyObject * THPVariable_is_same_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_same_size(Tensor input, Tensor other)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::is_same_size(Tensor self, Tensor other) -> bool
  auto dispatch_is_same_size = [](const Tensor & self, const Tensor & other) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_same_size(other);
  };
  return wrap(dispatch_is_same_size(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_signed
static PyObject * THPVariable_is_signed(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_signed(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::is_signed(Tensor self) -> bool
  auto dispatch_is_signed = [](const Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_signed();
  };
  return wrap(dispatch_is_signed(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// isclose
static PyObject * THPVariable_isclose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isclose(Tensor input, Tensor other, double rtol=1e-05, double atol=1e-08, bool equal_nan=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor
  auto dispatch_isclose = [](const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.isclose(other, rtol, atol, equal_nan);
  };
  return wrap(dispatch_isclose(_r.tensor(0), _r.tensor(1), _r.toDouble(2), _r.toDouble(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// isfinite
static PyObject * THPVariable_isfinite(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isfinite(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::isfinite(Tensor self) -> Tensor
  auto dispatch_isfinite = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::isfinite(self);
  };
  return wrap(dispatch_isfinite(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// isinf
static PyObject * THPVariable_isinf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isinf(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::isinf(Tensor self) -> Tensor
  auto dispatch_isinf = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::isinf(self);
  };
  return wrap(dispatch_isinf(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// isnan
static PyObject * THPVariable_isnan(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isnan(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::isnan(Tensor self) -> Tensor
  auto dispatch_isnan = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::isnan(self);
  };
  return wrap(dispatch_isnan(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// kl_div
static PyObject * THPVariable_kl_div(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "kl_div(Tensor input, Tensor target, int64_t reduction=at::Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::kl_div(Tensor self, Tensor target, int reduction=Mean) -> Tensor
  auto dispatch_kl_div = [](const Tensor & self, const Tensor & target, int64_t reduction) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::kl_div(self, target, reduction);
  };
  return wrap(dispatch_kl_div(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
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
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.kthvalue_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "kthvalue(Tensor input, int64_t k, Dimname dim, bool keepdim=False, *, TensorList[2] out=None)",
    "kthvalue(Tensor input, int64_t k, int64_t dim=-1, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::kthvalue.dimname(Tensor self, int k, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        auto dispatch_kthvalue = [](const Tensor & self, int64_t k, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.kthvalue(k, dim, keepdim);
        };
        return wrap(&NamedTuple, dispatch_kthvalue(_r.tensor(0), _r.toInt64(1), _r.dimname(2), _r.toBool(3)));
      } else {
        // aten::kthvalue.dimname_out(Tensor self, int k, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(4);
        auto dispatch_kthvalue_out = [](Tensor & values, Tensor & indices, const Tensor & self, int64_t k, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::kthvalue_out(values, indices, self, k, dim, keepdim);
        };
        return wrap(&NamedTuple1, dispatch_kthvalue_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.dimname(2), _r.toBool(3)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
        auto dispatch_kthvalue = [](const Tensor & self, int64_t k, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.kthvalue(k, dim, keepdim);
        };
        return wrap(&NamedTuple, dispatch_kthvalue(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toBool(3)));
      } else {
        // aten::kthvalue.values(Tensor self, int k, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(4);
        auto dispatch_kthvalue_out = [](Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::kthvalue_out(values, indices, self, k, dim, keepdim);
        };
        return wrap(&NamedTuple1, dispatch_kthvalue_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toBool(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// layer_norm
static PyObject * THPVariable_layer_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "layer_norm(Tensor input, IntArrayRef normalized_shape, Tensor? weight=None, Tensor? bias=None, double eps=1e-05, bool cudnn_enable=True)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor
  auto dispatch_layer_norm = [](const Tensor & input, IntArrayRef normalized_shape, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enable) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable);
  };
  return wrap(dispatch_layer_norm(_r.tensor(0), _r.intlist(1), _r.tensor(2), _r.tensor(3), _r.toDouble(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// le
static PyObject * THPVariable_le(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "le(Tensor input, Tensor other, *, Tensor out=None)",
    "le(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::le.Tensor(Tensor self, Tensor other) -> Tensor
        auto dispatch_le = [](const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.le(other);
        };
        return wrap(dispatch_le(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_le_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::le_out(out, self, other);
        };
        return wrap(dispatch_le_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::le.Scalar(Tensor self, Scalar other) -> Tensor
        auto dispatch_le = [](const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.le(other);
        };
        return wrap(dispatch_le(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_le_out = [](Tensor out, const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::le_out(out, self, other);
        };
        return wrap(dispatch_le_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
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
  static PythonArgParser parser({
    "lerp(Tensor input, Tensor end, Tensor weight, *, Tensor out=None)",
    "lerp(Tensor input, Tensor end, Scalar weight, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor
        auto dispatch_lerp = [](const Tensor & self, const Tensor & end, const Tensor & weight) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.lerp(end, weight);
        };
        return wrap(dispatch_lerp(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
      } else {
        // aten::lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_lerp_out = [](Tensor out, const Tensor & self, const Tensor & end, const Tensor & weight) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::lerp_out(out, self, end, weight);
        };
        return wrap(dispatch_lerp_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.tensor(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor
        auto dispatch_lerp = [](const Tensor & self, const Tensor & end, Scalar weight) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.lerp(end, weight);
        };
        return wrap(dispatch_lerp(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
      } else {
        // aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_lerp_out = [](Tensor out, const Tensor & self, const Tensor & end, Scalar weight) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::lerp_out(out, self, end, weight);
        };
        return wrap(dispatch_lerp_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.scalar(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lgamma
static PyObject * THPVariable_lgamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lgamma(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::lgamma(Tensor self) -> Tensor
    auto dispatch_lgamma = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.lgamma();
    };
    return wrap(dispatch_lgamma(_r.tensor(0)));
  } else {
    // aten::lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_lgamma_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::lgamma_out(out, self);
    };
    return wrap(dispatch_lgamma_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linspace
static PyObject * THPVariable_linspace(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linspace(Scalar start, Scalar end, int64_t steps=100, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(3)) {
    // aten::linspace(Scalar start, Scalar end, int steps=100, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    const auto options = TensorOptions()
        .dtype(_r.scalartype(4))
        .device(_r.device(6))
        .layout(_r.layout(5).layout)
        .requires_grad(_r.toBool(8))
        .pinned_memory(_r.toBool(7));
    torch::utils::maybe_initialize_cuda(options);
    auto dispatch_linspace = [](Scalar start, Scalar end, int64_t steps, const TensorOptions & options) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return torch::linspace(start, end, steps, options);
    };
    return wrap(dispatch_linspace(_r.scalar(0), _r.scalar(1), _r.toInt64(2), options));
  } else {
    // aten::linspace.out(Scalar start, Scalar end, int steps=100, *, Tensor(a!) out) -> Tensor(a!)
    check_out_type_matches(_r.tensor(3), _r.scalartype(4), _r.isNone(4),
                           _r.layout(5), _r.isNone(5),
                           _r.device(6), _r.isNone(6));
    auto dispatch_linspace_out = [](Tensor out, Scalar start, Scalar end, int64_t steps) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linspace_out(out, start, end, steps);
    };
    return wrap(dispatch_linspace_out(_r.tensor(3), _r.scalar(0), _r.scalar(1), _r.toInt64(2)).set_requires_grad(_r.toBool(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log
static PyObject * THPVariable_log(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::log(Tensor self) -> Tensor
    auto dispatch_log = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.log();
    };
    return wrap(dispatch_log(_r.tensor(0)));
  } else {
    // aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_log_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::log_out(out, self);
    };
    return wrap(dispatch_log_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log10
static PyObject * THPVariable_log10(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log10(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::log10(Tensor self) -> Tensor
    auto dispatch_log10 = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.log10();
    };
    return wrap(dispatch_log10(_r.tensor(0)));
  } else {
    // aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_log10_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::log10_out(out, self);
    };
    return wrap(dispatch_log10_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log10_
static PyObject * THPVariable_log10_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log10_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::log10_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_log10_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log10_();
  };
  return wrap(dispatch_log10_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log1p
static PyObject * THPVariable_log1p(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log1p(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::log1p(Tensor self) -> Tensor
    auto dispatch_log1p = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.log1p();
    };
    return wrap(dispatch_log1p(_r.tensor(0)));
  } else {
    // aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_log1p_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::log1p_out(out, self);
    };
    return wrap(dispatch_log1p_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log1p_
static PyObject * THPVariable_log1p_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log1p_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::log1p_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_log1p_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log1p_();
  };
  return wrap(dispatch_log1p_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log2
static PyObject * THPVariable_log2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log2(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::log2(Tensor self) -> Tensor
    auto dispatch_log2 = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.log2();
    };
    return wrap(dispatch_log2(_r.tensor(0)));
  } else {
    // aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_log2_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::log2_out(out, self);
    };
    return wrap(dispatch_log2_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log2_
static PyObject * THPVariable_log2_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log2_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::log2_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_log2_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log2_();
  };
  return wrap(dispatch_log2_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log_
static PyObject * THPVariable_log_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::log_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_log_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log_();
  };
  return wrap(dispatch_log_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// log_softmax
static PyObject * THPVariable_log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log_softmax(Tensor input, Dimname dim, *, ScalarType? dtype=None)",
    "log_softmax(Tensor input, int64_t dim, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_log_softmax = [](const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.log_softmax(dim, dtype);
      };
      return wrap(dispatch_log_softmax(_r.tensor(0), _r.dimname(1), _r.scalartypeOptional(2)));
    }
    case 1: {
      // aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
      auto dispatch_log_softmax = [](const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.log_softmax(dim, dtype);
      };
      return wrap(dispatch_log_softmax(_r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logdet
static PyObject * THPVariable_logdet(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logdet(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::logdet(Tensor self) -> Tensor
  auto dispatch_logdet = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logdet();
  };
  return wrap(dispatch_logdet(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_and
static PyObject * THPVariable_logical_and(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logical_and(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::logical_and(Tensor self, Tensor other) -> Tensor
    auto dispatch_logical_and = [](const Tensor & self, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.logical_and(other);
    };
    return wrap(dispatch_logical_and(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::logical_and.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_logical_and_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::logical_and_out(out, self, other);
    };
    return wrap(dispatch_logical_and_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_not
static PyObject * THPVariable_logical_not(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logical_not(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::logical_not(Tensor self) -> Tensor
    auto dispatch_logical_not = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.logical_not();
    };
    return wrap(dispatch_logical_not(_r.tensor(0)));
  } else {
    // aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_logical_not_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::logical_not_out(out, self);
    };
    return wrap(dispatch_logical_not_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_or
static PyObject * THPVariable_logical_or(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logical_or(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::logical_or(Tensor self, Tensor other) -> Tensor
    auto dispatch_logical_or = [](const Tensor & self, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.logical_or(other);
    };
    return wrap(dispatch_logical_or(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::logical_or.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_logical_or_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::logical_or_out(out, self, other);
    };
    return wrap(dispatch_logical_or_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_xor
static PyObject * THPVariable_logical_xor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logical_xor(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::logical_xor(Tensor self, Tensor other) -> Tensor
    auto dispatch_logical_xor = [](const Tensor & self, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.logical_xor(other);
    };
    return wrap(dispatch_logical_xor(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::logical_xor.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_logical_xor_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::logical_xor_out(out, self, other);
    };
    return wrap(dispatch_logical_xor_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logspace
static PyObject * THPVariable_logspace(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logspace(Scalar start, Scalar end, int64_t steps=100, double base=10.0, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(4)) {
    // aten::logspace(Scalar start, Scalar end, int steps=100, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    const auto options = TensorOptions()
        .dtype(_r.scalartype(5))
        .device(_r.device(7))
        .layout(_r.layout(6).layout)
        .requires_grad(_r.toBool(9))
        .pinned_memory(_r.toBool(8));
    torch::utils::maybe_initialize_cuda(options);
    auto dispatch_logspace = [](Scalar start, Scalar end, int64_t steps, double base, const TensorOptions & options) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return torch::logspace(start, end, steps, base, options);
    };
    return wrap(dispatch_logspace(_r.scalar(0), _r.scalar(1), _r.toInt64(2), _r.toDouble(3), options));
  } else {
    // aten::logspace.out(Scalar start, Scalar end, int steps=100, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)
    check_out_type_matches(_r.tensor(4), _r.scalartype(5), _r.isNone(5),
                           _r.layout(6), _r.isNone(6),
                           _r.device(7), _r.isNone(7));
    auto dispatch_logspace_out = [](Tensor out, Scalar start, Scalar end, int64_t steps, double base) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::logspace_out(out, start, end, steps, base);
    };
    return wrap(dispatch_logspace_out(_r.tensor(4), _r.scalar(0), _r.scalar(1), _r.toInt64(2), _r.toDouble(3)).set_requires_grad(_r.toBool(9)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// logsumexp
static PyObject * THPVariable_logsumexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logsumexp(Tensor input, DimnameList[1] dim, bool keepdim=False, *, Tensor out=None)",
    "logsumexp(Tensor input, IntArrayRef[1] dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::logsumexp.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor
        auto dispatch_logsumexp = [](const Tensor & self, DimnameList dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.logsumexp(dim, keepdim);
        };
        return wrap(dispatch_logsumexp(_r.tensor(0), _r.dimnamelist(1), _r.toBool(2)));
      } else {
        // aten::logsumexp.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_logsumexp_out = [](Tensor out, const Tensor & self, DimnameList dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::logsumexp_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_logsumexp_out(_r.tensor(3), _r.tensor(0), _r.dimnamelist(1), _r.toBool(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
        auto dispatch_logsumexp = [](const Tensor & self, IntArrayRef dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.logsumexp(dim, keepdim);
        };
        return wrap(dispatch_logsumexp(_r.tensor(0), _r.intlist(1), _r.toBool(2)));
      } else {
        // aten::logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_logsumexp_out = [](Tensor out, const Tensor & self, IntArrayRef dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::logsumexp_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_logsumexp_out(_r.tensor(3), _r.tensor(0), _r.intlist(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// lstm
static PyObject * THPVariable_lstm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lstm(Tensor data, Tensor batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)",
    "lstm(Tensor input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)
      auto dispatch_lstm = [](const Tensor & data, const Tensor & batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) -> std::tuple<Tensor,Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
      };
      return wrap(dispatch_lstm(_r.tensor(0), _r.tensor(1), _r.tensorlist(2), _r.tensorlist(3), _r.toBool(4), _r.toInt64(5), _r.toDouble(6), _r.toBool(7), _r.toBool(8)));
    }
    case 1: {
      // aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)
      auto dispatch_lstm = [](const Tensor & input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> std::tuple<Tensor,Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
      };
      return wrap(dispatch_lstm(_r.tensor(0), _r.tensorlist(1), _r.tensorlist(2), _r.toBool(3), _r.toInt64(4), _r.toDouble(5), _r.toBool(6), _r.toBool(7), _r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lstm_cell
static PyObject * THPVariable_lstm_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lstm_cell(Tensor input, TensorList hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)
  auto dispatch_lstm_cell = [](const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  };
  return wrap(dispatch_lstm_cell(_r.tensor(0), _r.tensorlist(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5)));
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
    static PyStructSequence_Desc desc = { "torch.return_types.lstsq_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.lstsq", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "lstsq(Tensor input, Tensor A, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::lstsq(Tensor self, Tensor A) -> (Tensor solution, Tensor QR)
    auto dispatch_lstsq = [](const Tensor & self, const Tensor & A) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.lstsq(A);
    };
    return wrap(&NamedTuple1, dispatch_lstsq(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::lstsq.X(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_lstsq_out = [](Tensor & X, Tensor & qr, const Tensor & self, const Tensor & A) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::lstsq_out(X, qr, self, A);
    };
    return wrap(&NamedTuple, dispatch_lstsq_out(out[0], out[1], _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// lt
static PyObject * THPVariable_lt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lt(Tensor input, Tensor other, *, Tensor out=None)",
    "lt(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::lt.Tensor(Tensor self, Tensor other) -> Tensor
        auto dispatch_lt = [](const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.lt(other);
        };
        return wrap(dispatch_lt(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_lt_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::lt_out(out, self, other);
        };
        return wrap(dispatch_lt_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::lt.Scalar(Tensor self, Scalar other) -> Tensor
        auto dispatch_lt = [](const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.lt(other);
        };
        return wrap(dispatch_lt(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_lt_out = [](Tensor out, const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::lt_out(out, self, other);
        };
        return wrap(dispatch_lt_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lu_solve
static PyObject * THPVariable_lu_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lu_solve(Tensor input, Tensor LU_data, Tensor LU_pivots, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(3)) {
    // aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor
    auto dispatch_lu_solve = [](const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.lu_solve(LU_data, LU_pivots);
    };
    return wrap(dispatch_lu_solve(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  } else {
    // aten::lu_solve.out(Tensor self, Tensor LU_data, Tensor LU_pivots, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_lu_solve_out = [](Tensor out, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::lu_solve_out(out, self, LU_data, LU_pivots);
    };
    return wrap(dispatch_lu_solve_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// margin_ranking_loss
static PyObject * THPVariable_margin_ranking_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, double margin=0.0, int64_t reduction=at::Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor
  auto dispatch_margin_ranking_loss = [](const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::margin_ranking_loss(input1, input2, target, margin, reduction);
  };
  return wrap(dispatch_margin_ranking_loss(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toDouble(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// masked_fill
static PyObject * THPVariable_masked_fill(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_fill(Tensor input, Tensor mask, Tensor value)",
    "masked_fill(Tensor input, Tensor mask, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
      auto dispatch_masked_fill = [](const Tensor & self, const Tensor & mask, const Tensor & value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.masked_fill(mask, value);
      };
      return wrap(dispatch_masked_fill(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
      auto dispatch_masked_fill = [](const Tensor & self, const Tensor & mask, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.masked_fill(mask, value);
      };
      return wrap(dispatch_masked_fill(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// masked_scatter
static PyObject * THPVariable_masked_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_scatter(Tensor input, Tensor mask, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
  auto dispatch_masked_scatter = [](const Tensor & self, const Tensor & mask, const Tensor & source) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.masked_scatter(mask, source);
  };
  return wrap(dispatch_masked_scatter(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// masked_select
static PyObject * THPVariable_masked_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_select(Tensor input, Tensor mask, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::masked_select(Tensor self, Tensor mask) -> Tensor
    auto dispatch_masked_select = [](const Tensor & self, const Tensor & mask) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.masked_select(mask);
    };
    return wrap(dispatch_masked_select(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::masked_select.out(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_masked_select_out = [](Tensor out, const Tensor & self, const Tensor & mask) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::masked_select_out(out, self, mask);
    };
    return wrap(dispatch_masked_select_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// matmul
static PyObject * THPVariable_matmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "matmul(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::matmul(Tensor self, Tensor other) -> Tensor
    auto dispatch_matmul = [](const Tensor & self, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.matmul(other);
    };
    return wrap(dispatch_matmul(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_matmul_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::matmul_out(out, self, other);
    };
    return wrap(dispatch_matmul_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// matrix_power
static PyObject * THPVariable_matrix_power(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "matrix_power(Tensor input, int64_t n)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::matrix_power(Tensor self, int n) -> Tensor
  auto dispatch_matrix_power = [](const Tensor & self, int64_t n) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.matrix_power(n);
  };
  return wrap(dispatch_matrix_power(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// matrix_rank
static PyObject * THPVariable_matrix_rank(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "matrix_rank(Tensor input, bool symmetric=False)",
    "matrix_rank(Tensor input, double tol, bool symmetric=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::matrix_rank(Tensor self, bool symmetric=False) -> Tensor
      auto dispatch_matrix_rank = [](const Tensor & self, bool symmetric) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::matrix_rank(self, symmetric);
      };
      return wrap(dispatch_matrix_rank(_r.tensor(0), _r.toBool(1)));
    }
    case 1: {
      // aten::matrix_rank.tol(Tensor self, float tol, bool symmetric=False) -> Tensor
      auto dispatch_matrix_rank = [](const Tensor & self, double tol, bool symmetric) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::matrix_rank(self, tol, symmetric);
      };
      return wrap(dispatch_matrix_rank(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
    }
  }
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
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.max_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "max(Tensor input)",
    "max(Tensor input, Dimname dim, bool keepdim=False, *, TensorList[2] out=None)",
    "max(Tensor input, Tensor other, *, Tensor out=None)",
    "max(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::max(Tensor self) -> Tensor
      auto dispatch_max = [](const Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.max();
      };
      return wrap(dispatch_max(_r.tensor(0)));
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        auto dispatch_max = [](const Tensor & self, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.max(dim, keepdim);
        };
        return wrap(&NamedTuple, dispatch_max(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::max.names_dim_max(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_max_out = [](Tensor & max, Tensor & max_values, const Tensor & self, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::max_out(max, max_values, self, dim, keepdim);
        };
        return wrap(&NamedTuple1, dispatch_max_out(out[0], out[1], _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::max.other(Tensor self, Tensor other) -> Tensor
        auto dispatch_max = [](const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.max(other);
        };
        return wrap(dispatch_max(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_max_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::max_out(out, self, other);
        };
        return wrap(dispatch_max_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 3: {
      if (_r.isNone(3)) {
        // aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        auto dispatch_max = [](const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.max(dim, keepdim);
        };
        return wrap(&NamedTuple, dispatch_max(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_max_out = [](Tensor & max, Tensor & max_values, const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::max_out(max, max_values, self, dim, keepdim);
        };
        return wrap(&NamedTuple1, dispatch_max_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_pool1d
static PyObject * THPVariable_max_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool1d(Tensor input, IntArrayRef[1] kernel_size, IntArrayRef[1] stride=None, IntArrayRef[1] padding=0, IntArrayRef[1] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor
  auto dispatch_max_pool1d = [](const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_max_pool1d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_pool1d_with_indices
static PyObject * THPVariable_max_pool1d_with_indices(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool1d_with_indices(Tensor input, IntArrayRef[1] kernel_size, IntArrayRef[1] stride=None, IntArrayRef[1] padding=0, IntArrayRef[1] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::max_pool1d_with_indices(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
  auto dispatch_max_pool1d_with_indices = [](const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::max_pool1d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_max_pool1d_with_indices(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_pool2d
static PyObject * THPVariable_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool2d(Tensor input, IntArrayRef[2] kernel_size, IntArrayRef[2] stride=None, IntArrayRef[2] padding=0, IntArrayRef[2] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
  auto dispatch_max_pool2d = [](const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_max_pool2d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_pool3d
static PyObject * THPVariable_max_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool3d(Tensor input, IntArrayRef[3] kernel_size, IntArrayRef[3] stride=None, IntArrayRef[3] padding=0, IntArrayRef[3] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
  auto dispatch_max_pool3d = [](const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_max_pool3d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// mean
static PyObject * THPVariable_mean(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mean(Tensor input, *, ScalarType? dtype=None)",
    "mean(Tensor input, DimnameList[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
    "mean(Tensor input, IntArrayRef[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_mean = [](const Tensor & self, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.mean(dtype);
      };
      return wrap(dispatch_mean(_r.tensor(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::mean.names_dim(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        auto dispatch_mean = [](const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.mean(dim, keepdim, dtype);
        };
        return wrap(dispatch_mean(_r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.scalartypeOptional(3)));
      } else {
        // aten::mean.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_mean_out = [](Tensor out, const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::mean_out(out, self, dim, keepdim, dtype);
        };
        return wrap(dispatch_mean_out(_r.tensor(4), _r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.scalartypeOptional(3)));
      }
    }
    case 2: {
      if (_r.isNone(4)) {
        // aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        auto dispatch_mean = [](const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.mean(dim, keepdim, dtype);
        };
        return wrap(dispatch_mean(_r.tensor(0), _r.intlist(1), _r.toBool(2), _r.scalartypeOptional(3)));
      } else {
        // aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_mean_out = [](Tensor out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::mean_out(out, self, dim, keepdim, dtype);
        };
        return wrap(dispatch_mean_out(_r.tensor(4), _r.tensor(0), _r.intlist(1), _r.toBool(2), _r.scalartypeOptional(3)));
      }
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
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.median_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "median(Tensor input)",
    "median(Tensor input, Dimname dim, bool keepdim=False, *, TensorList[2] out=None)",
    "median(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::median(Tensor self) -> Tensor
      auto dispatch_median = [](const Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.median();
      };
      return wrap(dispatch_median(_r.tensor(0)));
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::median.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        auto dispatch_median = [](const Tensor & self, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.median(dim, keepdim);
        };
        return wrap(&NamedTuple, dispatch_median(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::median.names_dim_values(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_median_out = [](Tensor & values, Tensor & indices, const Tensor & self, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::median_out(values, indices, self, dim, keepdim);
        };
        return wrap(&NamedTuple1, dispatch_median_out(out[0], out[1], _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        auto dispatch_median = [](const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.median(dim, keepdim);
        };
        return wrap(&NamedTuple, dispatch_median(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::median.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_median_out = [](Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::median_out(values, indices, self, dim, keepdim);
        };
        return wrap(&NamedTuple1, dispatch_median_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// meshgrid
static PyObject * THPVariable_meshgrid(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "meshgrid(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::meshgrid(Tensor[] tensors) -> Tensor[]
  auto dispatch_meshgrid = [](TensorList tensors) -> std::vector<Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::meshgrid(tensors);
  };
  return wrap(dispatch_meshgrid(_r.tensorlist(0)));
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
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.min_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "min(Tensor input)",
    "min(Tensor input, Dimname dim, bool keepdim=False, *, TensorList[2] out=None)",
    "min(Tensor input, Tensor other, *, Tensor out=None)",
    "min(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::min(Tensor self) -> Tensor
      auto dispatch_min = [](const Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.min();
      };
      return wrap(dispatch_min(_r.tensor(0)));
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::min.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        auto dispatch_min = [](const Tensor & self, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.min(dim, keepdim);
        };
        return wrap(&NamedTuple, dispatch_min(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::min.names_dim_min(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_min_out = [](Tensor & min, Tensor & min_indices, const Tensor & self, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::min_out(min, min_indices, self, dim, keepdim);
        };
        return wrap(&NamedTuple1, dispatch_min_out(out[0], out[1], _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::min.other(Tensor self, Tensor other) -> Tensor
        auto dispatch_min = [](const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.min(other);
        };
        return wrap(dispatch_min(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_min_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::min_out(out, self, other);
        };
        return wrap(dispatch_min_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 3: {
      if (_r.isNone(3)) {
        // aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        auto dispatch_min = [](const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.min(dim, keepdim);
        };
        return wrap(&NamedTuple, dispatch_min(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_min_out = [](Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::min_out(min, min_indices, self, dim, keepdim);
        };
        return wrap(&NamedTuple1, dispatch_min_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// miopen_batch_norm
static PyObject * THPVariable_miopen_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double exponential_average_factor, double epsilon)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)
  auto dispatch_miopen_batch_norm = [](const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) -> std::tuple<Tensor,Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::miopen_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
  };
  return wrap(dispatch_miopen_batch_norm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// miopen_convolution
static PyObject * THPVariable_miopen_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_convolution(Tensor input, Tensor weight, Tensor? bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::miopen_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
  auto dispatch_miopen_convolution = [](const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::miopen_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
  };
  return wrap(dispatch_miopen_convolution(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toInt64(6), _r.toBool(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// miopen_convolution_transpose
static PyObject * THPVariable_miopen_convolution_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_convolution_transpose(Tensor input, Tensor weight, Tensor? bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::miopen_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
  auto dispatch_miopen_convolution_transpose = [](const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::miopen_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  };
  return wrap(dispatch_miopen_convolution_transpose(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.intlist(6), _r.toInt64(7), _r.toBool(8), _r.toBool(9)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// miopen_depthwise_convolution
static PyObject * THPVariable_miopen_depthwise_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_depthwise_convolution(Tensor input, Tensor weight, Tensor? bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::miopen_depthwise_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
  auto dispatch_miopen_depthwise_convolution = [](const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::miopen_depthwise_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
  };
  return wrap(dispatch_miopen_depthwise_convolution(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toInt64(6), _r.toBool(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// miopen_rnn
static PyObject * THPVariable_miopen_rnn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_rnn(Tensor input, TensorList weight, int64_t weight_stride0, Tensor hx, Tensor? cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, Tensor? dropout_state)",
  }, /*traceable=*/true);

  ParsedArgs<14> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::miopen_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
  auto dispatch_miopen_rnn = [](const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state) -> std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::miopen_rnn(input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
  };
  return wrap(dispatch_miopen_rnn(_r.tensor(0), _r.tensorlist(1), _r.toInt64(2), _r.tensor(3), _r.tensor(4), _r.toInt64(5), _r.toInt64(6), _r.toInt64(7), _r.toBool(8), _r.toDouble(9), _r.toBool(10), _r.toBool(11), _r.intlist(12), _r.tensor(13)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_adaptive_avg_pool2d
static PyObject * THPVariable_mkldnn_adaptive_avg_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_adaptive_avg_pool2d(Tensor input, IntArrayRef[2] output_size)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::mkldnn_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
  auto dispatch_mkldnn_adaptive_avg_pool2d = [](const Tensor & self, IntArrayRef output_size) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::mkldnn_adaptive_avg_pool2d(self, output_size);
  };
  return wrap(dispatch_mkldnn_adaptive_avg_pool2d(_r.tensor(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_convolution
static PyObject * THPVariable_mkldnn_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_convolution(Tensor input, Tensor weight, Tensor? bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups) -> Tensor
  auto dispatch_mkldnn_convolution = [](const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::mkldnn_convolution(self, weight, bias, padding, stride, dilation, groups);
  };
  return wrap(dispatch_mkldnn_convolution(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toInt64(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_convolution_backward_weights
static PyObject * THPVariable_mkldnn_convolution_backward_weights(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_convolution_backward_weights(IntArrayRef weight_size, Tensor grad_output, Tensor input, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::mkldnn_convolution_backward_weights(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> (Tensor, Tensor)
  auto dispatch_mkldnn_convolution_backward_weights = [](IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::mkldnn_convolution_backward_weights(weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
  };
  return wrap(dispatch_mkldnn_convolution_backward_weights(_r.intlist(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.toInt64(6), _r.toBool(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_max_pool2d
static PyObject * THPVariable_mkldnn_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_max_pool2d(Tensor input, IntArrayRef[2] kernel_size, IntArrayRef[2] stride=None, IntArrayRef[2] padding=0, IntArrayRef[2] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::mkldnn_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
  auto dispatch_mkldnn_max_pool2d = [](const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::mkldnn_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_mkldnn_max_pool2d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mm
static PyObject * THPVariable_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mm(Tensor input, Tensor mat2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::mm(Tensor self, Tensor mat2) -> Tensor
    auto dispatch_mm = [](const Tensor & self, const Tensor & mat2) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.mm(mat2);
    };
    return wrap(dispatch_mm(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_mm_out = [](Tensor out, const Tensor & self, const Tensor & mat2) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mm_out(out, self, mat2);
    };
    return wrap(dispatch_mm_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
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
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.mode_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "mode(Tensor input, Dimname dim, bool keepdim=False, *, TensorList[2] out=None)",
    "mode(Tensor input, int64_t dim=-1, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::mode.dimname(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        auto dispatch_mode = [](const Tensor & self, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.mode(dim, keepdim);
        };
        return wrap(&NamedTuple, dispatch_mode(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::mode.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_mode_out = [](Tensor & values, Tensor & indices, const Tensor & self, Dimname dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::mode_out(values, indices, self, dim, keepdim);
        };
        return wrap(&NamedTuple1, dispatch_mode_out(out[0], out[1], _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
        auto dispatch_mode = [](const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.mode(dim, keepdim);
        };
        return wrap(&NamedTuple, dispatch_mode(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::mode.values(Tensor self, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_mode_out = [](Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::mode_out(values, indices, self, dim, keepdim);
        };
        return wrap(&NamedTuple1, dispatch_mode_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mul
static PyObject * THPVariable_mul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mul(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
    auto dispatch_mul = [](const Tensor & self, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.mul(other);
    };
    return wrap(dispatch_mul(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_mul_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mul_out(out, self, other);
    };
    return wrap(dispatch_mul_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// multinomial
static PyObject * THPVariable_multinomial(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "multinomial(Tensor input, int64_t num_samples, bool replacement=False, *, Generator generator=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(4)) {
    // aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor
    auto dispatch_multinomial = [](const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.multinomial(num_samples, replacement, generator);
    };
    return wrap(dispatch_multinomial(_r.tensor(0), _r.toInt64(1), _r.toBool(2), _r.generator(3)));
  } else {
    // aten::multinomial.out(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_multinomial_out = [](Tensor out, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::multinomial_out(out, self, num_samples, replacement, generator);
    };
    return wrap(dispatch_multinomial_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.toBool(2), _r.generator(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mv
static PyObject * THPVariable_mv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mv(Tensor input, Tensor vec, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::mv(Tensor self, Tensor vec) -> Tensor
    auto dispatch_mv = [](const Tensor & self, const Tensor & vec) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.mv(vec);
    };
    return wrap(dispatch_mv(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::mv.out(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_mv_out = [](Tensor out, const Tensor & self, const Tensor & vec) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mv_out(out, self, vec);
    };
    return wrap(dispatch_mv_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mvlgamma
static PyObject * THPVariable_mvlgamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mvlgamma(Tensor input, int64_t p)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::mvlgamma(Tensor self, int p) -> Tensor
  auto dispatch_mvlgamma = [](const Tensor & self, int64_t p) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.mvlgamma(p);
  };
  return wrap(dispatch_mvlgamma(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// narrow
static PyObject * THPVariable_narrow(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "narrow(Tensor input, int64_t dim, int64_t start, int64_t length)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)
  auto dispatch_narrow = [](const Tensor & self, int64_t dim, int64_t start, int64_t length) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.narrow(dim, start, length);
  };
  return wrap(dispatch_narrow(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toInt64(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// native_batch_norm
static PyObject * THPVariable_native_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double momentum, double eps, *, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(8)) {
    // aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
    auto dispatch_native_batch_norm = [](const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) -> std::tuple<Tensor,Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
    };
    return wrap(dispatch_native_batch_norm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7)));
  } else {
    // aten::native_batch_norm.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))
    auto out = _r.tensorlist_n<3>(8);
    auto dispatch_native_batch_norm_out = [](Tensor & out, Tensor & save_mean, Tensor & save_invstd, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) -> std::tuple<Tensor,Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::native_batch_norm_out(out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
    };
    return wrap(dispatch_native_batch_norm_out(out[0], out[1], out[2], _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// native_layer_norm
static PyObject * THPVariable_native_layer_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_layer_norm(Tensor input, Tensor? weight, Tensor? bias, int64_t M, int64_t N, double eps)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::native_layer_norm(Tensor input, Tensor? weight, Tensor? bias, int M, int N, float eps) -> (Tensor, Tensor, Tensor)
  auto dispatch_native_layer_norm = [](const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t M, int64_t N, double eps) -> std::tuple<Tensor,Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::native_layer_norm(input, weight, bias, M, N, eps);
  };
  return wrap(dispatch_native_layer_norm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4), _r.toDouble(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// native_norm
static PyObject * THPVariable_native_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_norm(Tensor input, Scalar p=2)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::native_norm(Tensor self, Scalar p=2) -> Tensor
  auto dispatch_native_norm = [](const Tensor & self, Scalar p) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::native_norm(self, p);
  };
  return wrap(dispatch_native_norm(_r.tensor(0), _r.scalar(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ne
static PyObject * THPVariable_ne(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ne(Tensor input, Tensor other, *, Tensor out=None)",
    "ne(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::ne.Tensor(Tensor self, Tensor other) -> Tensor
        auto dispatch_ne = [](const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.ne(other);
        };
        return wrap(dispatch_ne(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_ne_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::ne_out(out, self, other);
        };
        return wrap(dispatch_ne_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::ne.Scalar(Tensor self, Scalar other) -> Tensor
        auto dispatch_ne = [](const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.ne(other);
        };
        return wrap(dispatch_ne(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_ne_out = [](Tensor out, const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::ne_out(out, self, other);
        };
        return wrap(dispatch_ne_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// neg
static PyObject * THPVariable_neg(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "neg(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::neg(Tensor self) -> Tensor
    auto dispatch_neg = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.neg();
    };
    return wrap(dispatch_neg(_r.tensor(0)));
  } else {
    // aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_neg_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::neg_out(out, self);
    };
    return wrap(dispatch_neg_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// neg_
static PyObject * THPVariable_neg_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "neg_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::neg_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_neg_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.neg_();
  };
  return wrap(dispatch_neg_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// norm
static PyObject * THPVariable_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "norm(Tensor input, Scalar p=2)",
    "norm(Tensor input, Scalar? p, *, ScalarType dtype)",
    "norm(Tensor input, Scalar? p, DimnameList[1] dim, bool keepdim, *, ScalarType dtype, Tensor out=None)",
    "norm(Tensor input, Scalar? p, DimnameList[1] dim, bool keepdim=False, *, Tensor out=None)",
    "norm(Tensor input, Scalar? p, IntArrayRef[1] dim, bool keepdim, *, ScalarType dtype, Tensor out=None)",
    "norm(Tensor input, Scalar? p, IntArrayRef[1] dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor
      auto dispatch_norm = [](const Tensor & self, Scalar p) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p);
      };
      return wrap(dispatch_norm(_r.tensor(0), _r.scalar(1)));
    }
    case 1: {
      // aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor
      auto dispatch_norm = [](const Tensor & self, c10::optional<Scalar> p, ScalarType dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p, dtype);
      };
      return wrap(dispatch_norm(_r.tensor(0), _r.scalarOptional(1), _r.scalartype(2)));
    }
    case 2: {
      if (_r.isNone(5)) {
        // aten::norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
        auto dispatch_norm = [](const Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim, ScalarType dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.norm(p, dim, keepdim, dtype);
        };
        return wrap(dispatch_norm(_r.tensor(0), _r.scalarOptional(1), _r.dimnamelist(2), _r.toBool(3), _r.scalartype(4)));
      } else {
        // aten::norm.names_dtype_out(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_norm_out = [](Tensor out, const Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim, ScalarType dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::norm_out(out, self, p, dim, keepdim, dtype);
        };
        return wrap(dispatch_norm_out(_r.tensor(5), _r.tensor(0), _r.scalarOptional(1), _r.dimnamelist(2), _r.toBool(3), _r.scalartype(4)));
      }
    }
    case 3: {
      if (_r.isNone(4)) {
        // aten::norm.names_ScalarOpt_dim(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False) -> Tensor
        auto dispatch_norm = [](const Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.norm(p, dim, keepdim);
        };
        return wrap(dispatch_norm(_r.tensor(0), _r.scalarOptional(1), _r.dimnamelist(2), _r.toBool(3)));
      } else {
        // aten::norm.names_out(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_norm_out = [](Tensor out, const Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::norm_out(out, self, p, dim, keepdim);
        };
        return wrap(dispatch_norm_out(_r.tensor(4), _r.tensor(0), _r.scalarOptional(1), _r.dimnamelist(2), _r.toBool(3)));
      }
    }
    case 4: {
      if (_r.isNone(5)) {
        // aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
        auto dispatch_norm = [](const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.norm(p, dim, keepdim, dtype);
        };
        return wrap(dispatch_norm(_r.tensor(0), _r.scalarOptional(1), _r.intlist(2), _r.toBool(3), _r.scalartype(4)));
      } else {
        // aten::norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_norm_out = [](Tensor out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::norm_out(out, self, p, dim, keepdim, dtype);
        };
        return wrap(dispatch_norm_out(_r.tensor(5), _r.tensor(0), _r.scalarOptional(1), _r.intlist(2), _r.toBool(3), _r.scalartype(4)));
      }
    }
    case 5: {
      if (_r.isNone(4)) {
        // aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor
        auto dispatch_norm = [](const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.norm(p, dim, keepdim);
        };
        return wrap(dispatch_norm(_r.tensor(0), _r.scalarOptional(1), _r.intlist(2), _r.toBool(3)));
      } else {
        // aten::norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_norm_out = [](Tensor out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::norm_out(out, self, p, dim, keepdim);
        };
        return wrap(dispatch_norm_out(_r.tensor(4), _r.tensor(0), _r.scalarOptional(1), _r.intlist(2), _r.toBool(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// norm_except_dim
static PyObject * THPVariable_norm_except_dim(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "norm_except_dim(Tensor v, int64_t pow=2, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::norm_except_dim(Tensor v, int pow=2, int dim=0) -> Tensor
  auto dispatch_norm_except_dim = [](const Tensor & v, int64_t pow, int64_t dim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::norm_except_dim(v, pow, dim);
  };
  return wrap(dispatch_norm_except_dim(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// normal
static PyObject * THPVariable_normal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "normal(Tensor mean, Tensor std, *, Generator generator=None, Tensor out=None)",
    "normal(Tensor mean, double std=1, *, Generator generator=None, Tensor out=None)",
    "normal(double mean, Tensor std, *, Generator generator=None, Tensor out=None)",
    "normal(double mean, double std, IntArrayRef size, *, Generator generator=None, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor
        auto dispatch_normal = [](const Tensor & mean, const Tensor & std, Generator * generator) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::normal(mean, std, generator);
        };
        return wrap(dispatch_normal(_r.tensor(0), _r.tensor(1), _r.generator(2)));
      } else {
        // aten::normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_normal_out = [](Tensor out, const Tensor & mean, const Tensor & std, Generator * generator) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::normal_out(out, mean, std, generator);
        };
        return wrap(dispatch_normal_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.generator(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor
        auto dispatch_normal = [](const Tensor & mean, double std, Generator * generator) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::normal(mean, std, generator);
        };
        return wrap(dispatch_normal(_r.tensor(0), _r.toDouble(1), _r.generator(2)));
      } else {
        // aten::normal.Tensor_float_out(Tensor mean, float std=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_normal_out = [](Tensor out, const Tensor & mean, double std, Generator * generator) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::normal_out(out, mean, std, generator);
        };
        return wrap(dispatch_normal_out(_r.tensor(3), _r.tensor(0), _r.toDouble(1), _r.generator(2)));
      }
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor
        auto dispatch_normal = [](double mean, const Tensor & std, Generator * generator) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::normal(mean, std, generator);
        };
        return wrap(dispatch_normal(_r.toDouble(0), _r.tensor(1), _r.generator(2)));
      } else {
        // aten::normal.float_Tensor_out(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_normal_out = [](Tensor out, double mean, const Tensor & std, Generator * generator) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::normal_out(out, mean, std, generator);
        };
        return wrap(dispatch_normal_out(_r.tensor(3), _r.toDouble(0), _r.tensor(1), _r.generator(2)));
      }
    }
    case 3: {
      if (_r.isNone(4)) {
        // aten::normal.float_float(float mean, float std, int[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartype(5))
            .device(_r.device(7))
            .layout(_r.layout(6).layout)
            .requires_grad(_r.toBool(9))
            .pinned_memory(_r.toBool(8));
        torch::utils::maybe_initialize_cuda(options);
        auto dispatch_normal = [](double mean, double std, IntArrayRef size, Generator * generator, const TensorOptions & options) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::normal(mean, std, size, generator, options);
        };
        return wrap(dispatch_normal(_r.toDouble(0), _r.toDouble(1), _r.intlist(2), _r.generator(3), options));
      } else {
        // aten::normal.float_float_out(float mean, float std, int[] size, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(4), _r.scalartype(5), _r.isNone(5),
                               _r.layout(6), _r.isNone(6),
                               _r.device(7), _r.isNone(7));
        auto dispatch_normal_out = [](Tensor out, double mean, double std, IntArrayRef size, Generator * generator) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::normal_out(out, mean, std, size, generator);
        };
        return wrap(dispatch_normal_out(_r.tensor(4), _r.toDouble(0), _r.toDouble(1), _r.intlist(2), _r.generator(3)).set_requires_grad(_r.toBool(9)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// nuclear_norm
static PyObject * THPVariable_nuclear_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nuclear_norm(Tensor input, IntArrayRef[2] dim, bool keepdim=False, *, Tensor out=None)",
    "nuclear_norm(Tensor input, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::nuclear_norm.dim(Tensor self, int[2] dim, bool keepdim=False) -> Tensor
        auto dispatch_nuclear_norm = [](const Tensor & self, IntArrayRef dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::nuclear_norm(self, dim, keepdim);
        };
        return wrap(dispatch_nuclear_norm(_r.tensor(0), _r.intlist(1), _r.toBool(2)));
      } else {
        // aten::nuclear_norm.dim_out(Tensor self, int[2] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_nuclear_norm_out = [](Tensor out, const Tensor & self, IntArrayRef dim, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::nuclear_norm_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_nuclear_norm_out(_r.tensor(3), _r.tensor(0), _r.intlist(1), _r.toBool(2)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::nuclear_norm(Tensor self, bool keepdim=False) -> Tensor
        auto dispatch_nuclear_norm = [](const Tensor & self, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::nuclear_norm(self, keepdim);
        };
        return wrap(dispatch_nuclear_norm(_r.tensor(0), _r.toBool(1)));
      } else {
        // aten::nuclear_norm.out(Tensor self, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_nuclear_norm_out = [](Tensor out, const Tensor & self, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::nuclear_norm_out(out, self, keepdim);
        };
        return wrap(dispatch_nuclear_norm_out(_r.tensor(2), _r.tensor(0), _r.toBool(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ones
static PyObject * THPVariable_ones(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ones(IntArrayRef size, *, DimnameList? names, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "ones(IntArrayRef size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::ones.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(1);
      c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartype(2))
          .device(_r.device(4))
          .layout(_r.layout(3).layout)
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_ones = [](IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::ones(size, names, options);
      };
      return wrap(dispatch_ones(_r.intlist(0), names, options));
    }
    case 1: {
      if (_r.isNone(1)) {
        // aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartype(2))
            .device(_r.device(4))
            .layout(_r.layout(3).layout)
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch::utils::maybe_initialize_cuda(options);
        auto dispatch_ones = [](IntArrayRef size, const TensorOptions & options) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::ones(size, options);
        };
        return wrap(dispatch_ones(_r.intlist(0), options));
      } else {
        // aten::ones.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartype(2), _r.isNone(2),
                               _r.layout(3), _r.isNone(3),
                               _r.device(4), _r.isNone(4));
        auto dispatch_ones_out = [](Tensor out, IntArrayRef size) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::ones_out(out, size);
        };
        return wrap(dispatch_ones_out(_r.tensor(1), _r.intlist(0)).set_requires_grad(_r.toBool(6)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ones_like
static PyObject * THPVariable_ones_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ones_like(Tensor input, *, MemoryFormat? memory_format=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "ones_like(Tensor input, *, MemoryFormat? memory_format=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::ones_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor
      auto self = _r.tensor(0);
      const auto options = TensorOptions()
          .dtype(_r.scalartypeWithDefault(2, self.scalar_type()))
          .device(_r.deviceWithDefault(4, self.device()))
          .layout(_r.layoutWithDefault(3, *torch::getLayout(self.options().backend())).layout)
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_ones_like = [](const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::ones_like(self, options, memory_format);
      };
      return wrap(dispatch_ones_like(self, options, _r.memoryformatOptional(1)));
    }
    case 1: {
      // aten::ones_like(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
      auto dispatch_ones_like = [](const Tensor & self, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::ones_like(self, memory_format);
      };
      return wrap(dispatch_ones_like(_r.tensor(0), _r.memoryformatOptional(1)).set_requires_grad(_r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// orgqr
static PyObject * THPVariable_orgqr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "orgqr(Tensor input, Tensor input2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::orgqr(Tensor self, Tensor input2) -> Tensor
    auto dispatch_orgqr = [](const Tensor & self, const Tensor & input2) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.orgqr(input2);
    };
    return wrap(dispatch_orgqr(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::orgqr.out(Tensor self, Tensor input2, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_orgqr_out = [](Tensor out, const Tensor & self, const Tensor & input2) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::orgqr_out(out, self, input2);
    };
    return wrap(dispatch_orgqr_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ormqr
static PyObject * THPVariable_ormqr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ormqr(Tensor input, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(5)) {
    // aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor
    auto dispatch_ormqr = [](const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.ormqr(input2, input3, left, transpose);
    };
    return wrap(dispatch_ormqr(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3), _r.toBool(4)));
  } else {
    // aten::ormqr.out(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_ormqr_out = [](Tensor out, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::ormqr_out(out, self, input2, input3, left, transpose);
    };
    return wrap(dispatch_ormqr_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3), _r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// pairwise_distance
static PyObject * THPVariable_pairwise_distance(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pairwise_distance(Tensor x1, Tensor x2, double p=2, double eps=1e-06, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::pairwise_distance(Tensor x1, Tensor x2, float p=2, float eps=1e-06, bool keepdim=False) -> Tensor
  auto dispatch_pairwise_distance = [](const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::pairwise_distance(x1, x2, p, eps, keepdim);
  };
  return wrap(dispatch_pairwise_distance(_r.tensor(0), _r.tensor(1), _r.toDouble(2), _r.toDouble(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// pdist
static PyObject * THPVariable_pdist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pdist(Tensor input, double p=2)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::pdist(Tensor self, float p=2) -> Tensor
  auto dispatch_pdist = [](const Tensor & self, double p) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::pdist(self, p);
  };
  return wrap(dispatch_pdist(_r.tensor(0), _r.toDouble(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// pinverse
static PyObject * THPVariable_pinverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pinverse(Tensor input, double rcond=1e-15)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::pinverse(Tensor self, float rcond=1e-15) -> Tensor
  auto dispatch_pinverse = [](const Tensor & self, double rcond) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.pinverse(rcond);
  };
  return wrap(dispatch_pinverse(_r.tensor(0), _r.toDouble(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// pixel_shuffle
static PyObject * THPVariable_pixel_shuffle(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pixel_shuffle(Tensor input, int64_t upscale_factor)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor
  auto dispatch_pixel_shuffle = [](const Tensor & self, int64_t upscale_factor) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::pixel_shuffle(self, upscale_factor);
  };
  return wrap(dispatch_pixel_shuffle(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// poisson
static PyObject * THPVariable_poisson(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "poisson(Tensor input, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::poisson(Tensor self, Generator? generator=None) -> Tensor
  auto dispatch_poisson = [](const Tensor & self, Generator * generator) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::poisson(self, generator);
  };
  return wrap(dispatch_poisson(_r.tensor(0), _r.generator(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// poisson_nll_loss
static PyObject * THPVariable_poisson_nll_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, double eps, int64_t reduction)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, float eps, int reduction) -> Tensor
  auto dispatch_poisson_nll_loss = [](const Tensor & input, const Tensor & target, bool log_input, bool full, double eps, int64_t reduction) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::poisson_nll_loss(input, target, log_input, full, eps, reduction);
  };
  return wrap(dispatch_poisson_nll_loss(_r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3), _r.toDouble(4), _r.toInt64(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// polygamma
static PyObject * THPVariable_polygamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "polygamma(int64_t n, Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::polygamma(int n, Tensor self) -> Tensor
    auto dispatch_polygamma = [](int64_t n, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.polygamma(n);
    };
    return wrap(dispatch_polygamma(_r.toInt64(0), _r.tensor(1)));
  } else {
    // aten::polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_polygamma_out = [](Tensor out, int64_t n, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::polygamma_out(out, n, self);
    };
    return wrap(dispatch_polygamma_out(_r.tensor(2), _r.toInt64(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// pow
static PyObject * THPVariable_pow(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pow(Tensor input, Tensor exponent, *, Tensor out=None)",
    "pow(Scalar self, Tensor exponent, *, Tensor out=None)",
    "pow(Tensor input, Scalar exponent, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
        auto dispatch_pow = [](const Tensor & self, const Tensor & exponent) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.pow(exponent);
        };
        return wrap(dispatch_pow(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_pow_out = [](Tensor out, const Tensor & self, const Tensor & exponent) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::pow_out(out, self, exponent);
        };
        return wrap(dispatch_pow_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor
        auto dispatch_pow = [](Scalar self, const Tensor & exponent) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::pow(self, exponent);
        };
        return wrap(dispatch_pow(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_pow_out = [](Tensor out, Scalar self, const Tensor & exponent) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::pow_out(out, self, exponent);
        };
        return wrap(dispatch_pow_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
        auto dispatch_pow = [](const Tensor & self, Scalar exponent) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.pow(exponent);
        };
        return wrap(dispatch_pow(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_pow_out = [](Tensor out, const Tensor & self, Scalar exponent) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::pow_out(out, self, exponent);
        };
        return wrap(dispatch_pow_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// prelu
static PyObject * THPVariable_prelu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "prelu(Tensor input, Tensor weight)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::prelu(Tensor self, Tensor weight) -> Tensor
  auto dispatch_prelu = [](const Tensor & self, const Tensor & weight) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.prelu(weight);
  };
  return wrap(dispatch_prelu(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// prod
static PyObject * THPVariable_prod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "prod(Tensor input, *, ScalarType? dtype=None)",
    "prod(Tensor input, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
    "prod(Tensor input, int64_t dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_prod = [](const Tensor & self, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.prod(dtype);
      };
      return wrap(dispatch_prod(_r.tensor(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::prod.dim_Dimname(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        auto dispatch_prod = [](const Tensor & self, Dimname dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.prod(dim, keepdim, dtype);
        };
        return wrap(dispatch_prod(_r.tensor(0), _r.dimname(1), _r.toBool(2), _r.scalartypeOptional(3)));
      } else {
        // aten::prod.Dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_prod_out = [](Tensor out, const Tensor & self, Dimname dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::prod_out(out, self, dim, keepdim, dtype);
        };
        return wrap(dispatch_prod_out(_r.tensor(4), _r.tensor(0), _r.dimname(1), _r.toBool(2), _r.scalartypeOptional(3)));
      }
    }
    case 2: {
      if (_r.isNone(4)) {
        // aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        auto dispatch_prod = [](const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.prod(dim, keepdim, dtype);
        };
        return wrap(dispatch_prod(_r.tensor(0), _r.toInt64(1), _r.toBool(2), _r.scalartypeOptional(3)));
      } else {
        // aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_prod_out = [](Tensor out, const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::prod_out(out, self, dim, keepdim, dtype);
        };
        return wrap(dispatch_prod_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.toBool(2), _r.scalartypeOptional(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// promote_types
static PyObject * THPVariable_promote_types(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "promote_types(ScalarType type1, ScalarType type2)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::promote_types(ScalarType type1, ScalarType type2) -> ScalarType
  auto dispatch_promote_types = [](ScalarType type1, ScalarType type2) -> ScalarType {
    pybind11::gil_scoped_release no_gil;
    return at::promote_types(type1, type2);
  };
  return wrap(dispatch_promote_types(_r.scalartype(0), _r.scalartype(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// q_per_channel_axis
static PyObject * THPVariable_q_per_channel_axis(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "q_per_channel_axis(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::q_per_channel_axis(Tensor self) -> int
  auto dispatch_q_per_channel_axis = [](const Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self.q_per_channel_axis();
  };
  return wrap(dispatch_q_per_channel_axis(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// q_per_channel_scales
static PyObject * THPVariable_q_per_channel_scales(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "q_per_channel_scales(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::q_per_channel_scales(Tensor self) -> Tensor
  auto dispatch_q_per_channel_scales = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.q_per_channel_scales();
  };
  return wrap(dispatch_q_per_channel_scales(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// q_per_channel_zero_points
static PyObject * THPVariable_q_per_channel_zero_points(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "q_per_channel_zero_points(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::q_per_channel_zero_points(Tensor self) -> Tensor
  auto dispatch_q_per_channel_zero_points = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.q_per_channel_zero_points();
  };
  return wrap(dispatch_q_per_channel_zero_points(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// q_scale
static PyObject * THPVariable_q_scale(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "q_scale(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::q_scale(Tensor self) -> float
  auto dispatch_q_scale = [](const Tensor & self) -> double {
    pybind11::gil_scoped_release no_gil;
    return self.q_scale();
  };
  return wrap(dispatch_q_scale(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// q_zero_point
static PyObject * THPVariable_q_zero_point(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "q_zero_point(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::q_zero_point(Tensor self) -> int
  auto dispatch_q_zero_point = [](const Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self.q_zero_point();
  };
  return wrap(dispatch_q_zero_point(_r.tensor(0)));
  Py_RETURN_NONE;
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
    static PyStructSequence_Desc desc = { "torch.return_types.qr_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.qr", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "qr(Tensor input, bool some=True, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)
    auto dispatch_qr = [](const Tensor & self, bool some) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.qr(some);
    };
    return wrap(&NamedTuple1, dispatch_qr(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::qr.Q(Tensor self, bool some=True, *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_qr_out = [](Tensor & Q, Tensor & R, const Tensor & self, bool some) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::qr_out(Q, R, self, some);
    };
    return wrap(&NamedTuple, dispatch_qr_out(out[0], out[1], _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantize_per_channel
static PyObject * THPVariable_quantize_per_channel(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int64_t axis, ScalarType dtype)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::quantize_per_channel(Tensor self, Tensor scales, Tensor zero_points, int axis, ScalarType dtype) -> Tensor
  auto dispatch_quantize_per_channel = [](const Tensor & self, const Tensor & scales, const Tensor & zero_points, int64_t axis, ScalarType dtype) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantize_per_channel(self, scales, zero_points, axis, dtype);
  };
  return wrap(dispatch_quantize_per_channel(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.scalartype(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantize_per_tensor
static PyObject * THPVariable_quantize_per_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantize_per_tensor(Tensor input, double scale, int64_t zero_point, ScalarType dtype)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor
  auto dispatch_quantize_per_tensor = [](const Tensor & self, double scale, int64_t zero_point, ScalarType dtype) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantize_per_tensor(self, scale, zero_point, dtype);
  };
  return wrap(dispatch_quantize_per_tensor(_r.tensor(0), _r.toDouble(1), _r.toInt64(2), _r.scalartype(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// quantized_gru
static PyObject * THPVariable_quantized_gru(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_gru(Tensor data, Tensor batch_sizes, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)",
    "quantized_gru(Tensor input, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::quantized_gru.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
      auto dispatch_quantized_gru = [](const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::quantized_gru(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
      };
      return wrap(dispatch_quantized_gru(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensorlist(3), _r.toBool(4), _r.toInt64(5), _r.toDouble(6), _r.toBool(7), _r.toBool(8)));
    }
    case 1: {
      // aten::quantized_gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
      auto dispatch_quantized_gru = [](const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::quantized_gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
      };
      return wrap(dispatch_quantized_gru(_r.tensor(0), _r.tensor(1), _r.tensorlist(2), _r.toBool(3), _r.toInt64(4), _r.toDouble(5), _r.toBool(6), _r.toBool(7), _r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantized_gru_cell
static PyObject * THPVariable_quantized_gru_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh)",
  }, /*traceable=*/true);

  ParsedArgs<14> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
  auto dispatch_quantized_gru_cell = [](const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantized_gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  };
  return wrap(dispatch_quantized_gru_cell(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.tensor(6), _r.tensor(7), _r.tensor(8), _r.tensor(9), _r.scalar(10), _r.scalar(11), _r.scalar(12), _r.scalar(13)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// quantized_lstm
static PyObject * THPVariable_quantized_lstm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_lstm(Tensor data, Tensor batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, *, ScalarType? dtype=None, bool use_dynamic=False)",
    "quantized_lstm(Tensor input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, *, ScalarType? dtype=None, bool use_dynamic=False)",
  }, /*traceable=*/true);

  ParsedArgs<11> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::quantized_lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)
      auto dispatch_quantized_lstm = [](const Tensor & data, const Tensor & batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, c10::optional<ScalarType> dtype, bool use_dynamic) -> std::tuple<Tensor,Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::quantized_lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional, dtype, use_dynamic);
      };
      return wrap(dispatch_quantized_lstm(_r.tensor(0), _r.tensor(1), _r.tensorlist(2), _r.tensorlist(3), _r.toBool(4), _r.toInt64(5), _r.toDouble(6), _r.toBool(7), _r.toBool(8), _r.scalartypeOptional(9), _r.toBool(10)));
    }
    case 1: {
      // aten::quantized_lstm(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)
      auto dispatch_quantized_lstm = [](const Tensor & input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, c10::optional<ScalarType> dtype, bool use_dynamic) -> std::tuple<Tensor,Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::quantized_lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first, dtype, use_dynamic);
      };
      return wrap(dispatch_quantized_lstm(_r.tensor(0), _r.tensorlist(1), _r.tensorlist(2), _r.toBool(3), _r.toInt64(4), _r.toDouble(5), _r.toBool(6), _r.toBool(7), _r.toBool(8), _r.scalartypeOptional(9), _r.toBool(10)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantized_lstm_cell
static PyObject * THPVariable_quantized_lstm_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_lstm_cell(Tensor input, TensorList hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh)",
  }, /*traceable=*/true);

  ParsedArgs<14> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::quantized_lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor, Tensor)
  auto dispatch_quantized_lstm_cell = [](const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::quantized_lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  };
  return wrap(dispatch_quantized_lstm_cell(_r.tensor(0), _r.tensorlist(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.tensor(6), _r.tensor(7), _r.tensor(8), _r.tensor(9), _r.scalar(10), _r.scalar(11), _r.scalar(12), _r.scalar(13)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantized_max_pool2d
static PyObject * THPVariable_quantized_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_max_pool2d(Tensor input, IntArrayRef[2] kernel_size, IntArrayRef[2] stride=None, IntArrayRef[2] padding=0, IntArrayRef[2] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::quantized_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
  auto dispatch_quantized_max_pool2d = [](const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantized_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_quantized_max_pool2d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantized_rnn_relu_cell
static PyObject * THPVariable_quantized_rnn_relu_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh)",
  }, /*traceable=*/true);

  ParsedArgs<14> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
  auto dispatch_quantized_rnn_relu_cell = [](const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantized_rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  };
  return wrap(dispatch_quantized_rnn_relu_cell(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.tensor(6), _r.tensor(7), _r.tensor(8), _r.tensor(9), _r.scalar(10), _r.scalar(11), _r.scalar(12), _r.scalar(13)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantized_rnn_tanh_cell
static PyObject * THPVariable_quantized_rnn_tanh_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh)",
  }, /*traceable=*/true);

  ParsedArgs<14> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
  auto dispatch_quantized_rnn_tanh_cell = [](const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantized_rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  };
  return wrap(dispatch_quantized_rnn_tanh_cell(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.tensor(6), _r.tensor(7), _r.tensor(8), _r.tensor(9), _r.scalar(10), _r.scalar(11), _r.scalar(12), _r.scalar(13)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// rand
static PyObject * THPVariable_rand(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rand(IntArrayRef size, *, DimnameList? names, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "rand(IntArrayRef size, *, Generator generator, DimnameList? names, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "rand(IntArrayRef size, *, Generator generator, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "rand(IntArrayRef size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::rand.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(1);
      c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartype(2))
          .device(_r.device(4))
          .layout(_r.layout(3).layout)
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_rand = [](IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::rand(size, names, options);
      };
      return wrap(dispatch_rand(_r.intlist(0), names, options));
    }
    case 1: {
      // aten::rand.generator_with_names(int[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(2);
      c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartype(3))
          .device(_r.device(5))
          .layout(_r.layout(4).layout)
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_rand = [](IntArrayRef size, Generator * generator, c10::optional<DimnameList> names, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::rand(size, generator, names, options);
      };
      return wrap(dispatch_rand(_r.intlist(0), _r.generator(1), names, options));
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::rand.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartype(3))
            .device(_r.device(5))
            .layout(_r.layout(4).layout)
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch::utils::maybe_initialize_cuda(options);
        auto dispatch_rand = [](IntArrayRef size, Generator * generator, const TensorOptions & options) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::rand(size, generator, options);
        };
        return wrap(dispatch_rand(_r.intlist(0), _r.generator(1), options));
      } else {
        // aten::rand.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartype(3), _r.isNone(3),
                               _r.layout(4), _r.isNone(4),
                               _r.device(5), _r.isNone(5));
        auto dispatch_rand_out = [](Tensor out, IntArrayRef size, Generator * generator) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::rand_out(out, size, generator);
        };
        return wrap(dispatch_rand_out(_r.tensor(2), _r.intlist(0), _r.generator(1)).set_requires_grad(_r.toBool(7)));
      }
    }
    case 3: {
      if (_r.isNone(1)) {
        // aten::rand(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartype(2))
            .device(_r.device(4))
            .layout(_r.layout(3).layout)
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch::utils::maybe_initialize_cuda(options);
        auto dispatch_rand = [](IntArrayRef size, const TensorOptions & options) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::rand(size, options);
        };
        return wrap(dispatch_rand(_r.intlist(0), options));
      } else {
        // aten::rand.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartype(2), _r.isNone(2),
                               _r.layout(3), _r.isNone(3),
                               _r.device(4), _r.isNone(4));
        auto dispatch_rand_out = [](Tensor out, IntArrayRef size) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::rand_out(out, size);
        };
        return wrap(dispatch_rand_out(_r.tensor(1), _r.intlist(0)).set_requires_grad(_r.toBool(6)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// rand_like
static PyObject * THPVariable_rand_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rand_like(Tensor input, *, MemoryFormat? memory_format=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "rand_like(Tensor input, *, MemoryFormat? memory_format=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::rand_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor
      auto self = _r.tensor(0);
      const auto options = TensorOptions()
          .dtype(_r.scalartypeWithDefault(2, self.scalar_type()))
          .device(_r.deviceWithDefault(4, self.device()))
          .layout(_r.layoutWithDefault(3, *torch::getLayout(self.options().backend())).layout)
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_rand_like = [](const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::rand_like(self, options, memory_format);
      };
      return wrap(dispatch_rand_like(self, options, _r.memoryformatOptional(1)));
    }
    case 1: {
      // aten::rand_like(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
      auto dispatch_rand_like = [](const Tensor & self, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::rand_like(self, memory_format);
      };
      return wrap(dispatch_rand_like(_r.tensor(0), _r.memoryformatOptional(1)).set_requires_grad(_r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// randint_like
static PyObject * THPVariable_randint_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randint_like(Tensor input, int64_t high, *, MemoryFormat? memory_format=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "randint_like(Tensor input, int64_t high, *, MemoryFormat? memory_format=None, bool requires_grad=False)",
    "randint_like(Tensor input, int64_t low, int64_t high, *, MemoryFormat? memory_format=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "randint_like(Tensor input, int64_t low, int64_t high, *, MemoryFormat? memory_format=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::randint_like.dtype(Tensor self, int high, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor
      auto self = _r.tensor(0);
      const auto options = TensorOptions()
          .dtype(_r.scalartypeWithDefault(3, self.scalar_type()))
          .device(_r.deviceWithDefault(5, self.device()))
          .layout(_r.layoutWithDefault(4, *torch::getLayout(self.options().backend())).layout)
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_randint_like = [](const Tensor & self, int64_t high, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::randint_like(self, high, options, memory_format);
      };
      return wrap(dispatch_randint_like(self, _r.toInt64(1), options, _r.memoryformatOptional(2)));
    }
    case 1: {
      // aten::randint_like(Tensor self, int high, *, MemoryFormat? memory_format=None) -> Tensor
      auto dispatch_randint_like = [](const Tensor & self, int64_t high, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::randint_like(self, high, memory_format);
      };
      return wrap(dispatch_randint_like(_r.tensor(0), _r.toInt64(1), _r.memoryformatOptional(2)).set_requires_grad(_r.toBool(3)));
    }
    case 2: {
      // aten::randint_like.low_dtype(Tensor self, int low, int high, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor
      auto self = _r.tensor(0);
      const auto options = TensorOptions()
          .dtype(_r.scalartypeWithDefault(4, self.scalar_type()))
          .device(_r.deviceWithDefault(6, self.device()))
          .layout(_r.layoutWithDefault(5, *torch::getLayout(self.options().backend())).layout)
          .requires_grad(_r.toBool(8))
          .pinned_memory(_r.toBool(7));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_randint_like = [](const Tensor & self, int64_t low, int64_t high, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::randint_like(self, low, high, options, memory_format);
      };
      return wrap(dispatch_randint_like(self, _r.toInt64(1), _r.toInt64(2), options, _r.memoryformatOptional(3)));
    }
    case 3: {
      // aten::randint_like.low(Tensor self, int low, int high, *, MemoryFormat? memory_format=None) -> Tensor
      auto dispatch_randint_like = [](const Tensor & self, int64_t low, int64_t high, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::randint_like(self, low, high, memory_format);
      };
      return wrap(dispatch_randint_like(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.memoryformatOptional(3)).set_requires_grad(_r.toBool(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// randn
static PyObject * THPVariable_randn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randn(IntArrayRef size, *, DimnameList? names, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "randn(IntArrayRef size, *, Generator generator, DimnameList? names, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "randn(IntArrayRef size, *, Generator generator, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "randn(IntArrayRef size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::randn.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(1);
      c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartype(2))
          .device(_r.device(4))
          .layout(_r.layout(3).layout)
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_randn = [](IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::randn(size, names, options);
      };
      return wrap(dispatch_randn(_r.intlist(0), names, options));
    }
    case 1: {
      // aten::randn.generator_with_names(int[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(2);
      c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartype(3))
          .device(_r.device(5))
          .layout(_r.layout(4).layout)
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_randn = [](IntArrayRef size, Generator * generator, c10::optional<DimnameList> names, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::randn(size, generator, names, options);
      };
      return wrap(dispatch_randn(_r.intlist(0), _r.generator(1), names, options));
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::randn.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartype(3))
            .device(_r.device(5))
            .layout(_r.layout(4).layout)
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch::utils::maybe_initialize_cuda(options);
        auto dispatch_randn = [](IntArrayRef size, Generator * generator, const TensorOptions & options) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::randn(size, generator, options);
        };
        return wrap(dispatch_randn(_r.intlist(0), _r.generator(1), options));
      } else {
        // aten::randn.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartype(3), _r.isNone(3),
                               _r.layout(4), _r.isNone(4),
                               _r.device(5), _r.isNone(5));
        auto dispatch_randn_out = [](Tensor out, IntArrayRef size, Generator * generator) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::randn_out(out, size, generator);
        };
        return wrap(dispatch_randn_out(_r.tensor(2), _r.intlist(0), _r.generator(1)).set_requires_grad(_r.toBool(7)));
      }
    }
    case 3: {
      if (_r.isNone(1)) {
        // aten::randn(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartype(2))
            .device(_r.device(4))
            .layout(_r.layout(3).layout)
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch::utils::maybe_initialize_cuda(options);
        auto dispatch_randn = [](IntArrayRef size, const TensorOptions & options) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::randn(size, options);
        };
        return wrap(dispatch_randn(_r.intlist(0), options));
      } else {
        // aten::randn.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartype(2), _r.isNone(2),
                               _r.layout(3), _r.isNone(3),
                               _r.device(4), _r.isNone(4));
        auto dispatch_randn_out = [](Tensor out, IntArrayRef size) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::randn_out(out, size);
        };
        return wrap(dispatch_randn_out(_r.tensor(1), _r.intlist(0)).set_requires_grad(_r.toBool(6)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// randn_like
static PyObject * THPVariable_randn_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randn_like(Tensor input, *, MemoryFormat? memory_format=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "randn_like(Tensor input, *, MemoryFormat? memory_format=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::randn_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor
      auto self = _r.tensor(0);
      const auto options = TensorOptions()
          .dtype(_r.scalartypeWithDefault(2, self.scalar_type()))
          .device(_r.deviceWithDefault(4, self.device()))
          .layout(_r.layoutWithDefault(3, *torch::getLayout(self.options().backend())).layout)
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_randn_like = [](const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::randn_like(self, options, memory_format);
      };
      return wrap(dispatch_randn_like(self, options, _r.memoryformatOptional(1)));
    }
    case 1: {
      // aten::randn_like(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
      auto dispatch_randn_like = [](const Tensor & self, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::randn_like(self, memory_format);
      };
      return wrap(dispatch_randn_like(_r.tensor(0), _r.memoryformatOptional(1)).set_requires_grad(_r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// randperm
static PyObject * THPVariable_randperm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randperm(int64_t n, *, Generator generator, Tensor out=None, ScalarType dtype=torch.int64, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "randperm(int64_t n, *, Tensor out=None, ScalarType dtype=torch.int64, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::randperm.generator(int n, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartype(3))
            .device(_r.device(5))
            .layout(_r.layout(4).layout)
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch::utils::maybe_initialize_cuda(options);
        auto dispatch_randperm = [](int64_t n, Generator * generator, const TensorOptions & options) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::randperm(n, generator, options);
        };
        return wrap(dispatch_randperm(_r.toInt64(0), _r.generator(1), options));
      } else {
        // aten::randperm.generator_out(int n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartype(3), _r.isNone(3),
                               _r.layout(4), _r.isNone(4),
                               _r.device(5), _r.isNone(5));
        auto dispatch_randperm_out = [](Tensor out, int64_t n, Generator * generator) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::randperm_out(out, n, generator);
        };
        return wrap(dispatch_randperm_out(_r.tensor(2), _r.toInt64(0), _r.generator(1)).set_requires_grad(_r.toBool(7)));
      }
    }
    case 1: {
      if (_r.isNone(1)) {
        // aten::randperm(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartype(2))
            .device(_r.device(4))
            .layout(_r.layout(3).layout)
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch::utils::maybe_initialize_cuda(options);
        auto dispatch_randperm = [](int64_t n, const TensorOptions & options) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::randperm(n, options);
        };
        return wrap(dispatch_randperm(_r.toInt64(0), options));
      } else {
        // aten::randperm.out(int n, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartype(2), _r.isNone(2),
                               _r.layout(3), _r.isNone(3),
                               _r.device(4), _r.isNone(4));
        auto dispatch_randperm_out = [](Tensor out, int64_t n) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::randperm_out(out, n);
        };
        return wrap(dispatch_randperm_out(_r.tensor(1), _r.toInt64(0)).set_requires_grad(_r.toBool(6)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// real
static PyObject * THPVariable_real(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "real(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::real(Tensor self) -> Tensor
    auto dispatch_real = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.real();
    };
    return wrap(dispatch_real(_r.tensor(0)));
  } else {
    // aten::real.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_real_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::real_out(out, self);
    };
    return wrap(dispatch_real_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reciprocal
static PyObject * THPVariable_reciprocal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reciprocal(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::reciprocal(Tensor self) -> Tensor
    auto dispatch_reciprocal = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.reciprocal();
    };
    return wrap(dispatch_reciprocal(_r.tensor(0)));
  } else {
    // aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_reciprocal_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::reciprocal_out(out, self);
    };
    return wrap(dispatch_reciprocal_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reciprocal_
static PyObject * THPVariable_reciprocal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reciprocal_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_reciprocal_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.reciprocal_();
  };
  return wrap(dispatch_reciprocal_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// relu
static PyObject * THPVariable_relu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "relu(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::relu(Tensor self) -> Tensor
  auto dispatch_relu = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.relu();
  };
  return wrap(dispatch_relu(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// relu_
static PyObject * THPVariable_relu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "relu_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::relu_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_relu_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.relu_();
  };
  return wrap(dispatch_relu_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// remainder
static PyObject * THPVariable_remainder(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "remainder(Tensor input, Tensor other, *, Tensor out=None)",
    "remainder(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor
        auto dispatch_remainder = [](const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.remainder(other);
        };
        return wrap(dispatch_remainder(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_remainder_out = [](Tensor out, const Tensor & self, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::remainder_out(out, self, other);
        };
        return wrap(dispatch_remainder_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor
        auto dispatch_remainder = [](const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.remainder(other);
        };
        return wrap(dispatch_remainder(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_remainder_out = [](Tensor out, const Tensor & self, Scalar other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::remainder_out(out, self, other);
        };
        return wrap(dispatch_remainder_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// renorm
static PyObject * THPVariable_renorm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "renorm(Tensor input, Scalar p, int64_t dim, Scalar maxnorm, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(4)) {
    // aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor
    auto dispatch_renorm = [](const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.renorm(p, dim, maxnorm);
    };
    return wrap(dispatch_renorm(_r.tensor(0), _r.scalar(1), _r.toInt64(2), _r.scalar(3)));
  } else {
    // aten::renorm.out(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_renorm_out = [](Tensor out, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::renorm_out(out, self, p, dim, maxnorm);
    };
    return wrap(dispatch_renorm_out(_r.tensor(4), _r.tensor(0), _r.scalar(1), _r.toInt64(2), _r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// repeat_interleave
static PyObject * THPVariable_repeat_interleave(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "repeat_interleave(Tensor input, Tensor repeats, int64_t? dim=None)",
    "repeat_interleave(Tensor input, int64_t repeats, int64_t? dim=None)",
    "repeat_interleave(Tensor repeats)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None) -> Tensor
      auto dispatch_repeat_interleave = [](const Tensor & self, const Tensor & repeats, c10::optional<int64_t> dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.repeat_interleave(repeats, dim);
      };
      return wrap(dispatch_repeat_interleave(_r.tensor(0), _r.tensor(1), _r.toInt64Optional(2)));
    }
    case 1: {
      // aten::repeat_interleave.self_int(Tensor self, int repeats, int? dim=None) -> Tensor
      auto dispatch_repeat_interleave = [](const Tensor & self, int64_t repeats, c10::optional<int64_t> dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.repeat_interleave(repeats, dim);
      };
      return wrap(dispatch_repeat_interleave(_r.tensor(0), _r.toInt64(1), _r.toInt64Optional(2)));
    }
    case 2: {
      // aten::repeat_interleave.Tensor(Tensor repeats) -> Tensor
      auto dispatch_repeat_interleave = [](const Tensor & repeats) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::repeat_interleave(repeats);
      };
      return wrap(dispatch_repeat_interleave(_r.tensor(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reshape
static PyObject * THPVariable_reshape(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reshape(Tensor input, IntArrayRef shape)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::reshape(Tensor self, int[] shape) -> Tensor
  auto dispatch_reshape = [](const Tensor & self, IntArrayRef shape) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.reshape(shape);
  };
  return wrap(dispatch_reshape(_r.tensor(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// resize_as_
static PyObject * THPVariable_resize_as_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "resize_as_(Tensor input, Tensor the_template, *, MemoryFormat? memory_format=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)
  auto dispatch_resize_as_ = [](Tensor self, const Tensor & the_template, c10::optional<MemoryFormat> memory_format) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.resize_as_(the_template, memory_format);
  };
  return wrap(dispatch_resize_as_(_r.tensor(0), _r.tensor(1), _r.memoryformatOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// result_type
static PyObject * THPVariable_result_type(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "result_type(Tensor tensor, Tensor other)",
    "result_type(Scalar scalar, Tensor tensor)",
    "result_type(Tensor tensor, Scalar other)",
    "result_type(Scalar scalar1, Scalar scalar2)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::result_type.Tensor(Tensor tensor, Tensor other) -> ScalarType
      auto dispatch_result_type = [](const Tensor & tensor, const Tensor & other) -> ScalarType {
        pybind11::gil_scoped_release no_gil;
        return at::result_type(tensor, other);
      };
      return wrap(dispatch_result_type(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::result_type.Scalar_Tensor(Scalar scalar, Tensor tensor) -> ScalarType
      auto dispatch_result_type = [](Scalar scalar, const Tensor & tensor) -> ScalarType {
        pybind11::gil_scoped_release no_gil;
        return at::result_type(scalar, tensor);
      };
      return wrap(dispatch_result_type(_r.scalar(0), _r.tensor(1)));
    }
    case 2: {
      // aten::result_type.Scalar(Tensor tensor, Scalar other) -> ScalarType
      auto dispatch_result_type = [](const Tensor & tensor, Scalar other) -> ScalarType {
        pybind11::gil_scoped_release no_gil;
        return at::result_type(tensor, other);
      };
      return wrap(dispatch_result_type(_r.tensor(0), _r.scalar(1)));
    }
    case 3: {
      // aten::result_type.Scalar_Scalar(Scalar scalar1, Scalar scalar2) -> ScalarType
      auto dispatch_result_type = [](Scalar scalar1, Scalar scalar2) -> ScalarType {
        pybind11::gil_scoped_release no_gil;
        return at::result_type(scalar1, scalar2);
      };
      return wrap(dispatch_result_type(_r.scalar(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rfft
static PyObject * THPVariable_rfft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rfft(Tensor input, int64_t signal_ndim, bool normalized=False, bool onesided=True)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::rfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True) -> Tensor
  auto dispatch_rfft = [](const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rfft(signal_ndim, normalized, onesided);
  };
  return wrap(dispatch_rfft(_r.tensor(0), _r.toInt64(1), _r.toBool(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// rnn_relu
static PyObject * THPVariable_rnn_relu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rnn_relu(Tensor data, Tensor batch_sizes, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)",
    "rnn_relu(Tensor input, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::rnn_relu.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
      auto dispatch_rnn_relu = [](const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::rnn_relu(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
      };
      return wrap(dispatch_rnn_relu(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensorlist(3), _r.toBool(4), _r.toInt64(5), _r.toDouble(6), _r.toBool(7), _r.toBool(8)));
    }
    case 1: {
      // aten::rnn_relu.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
      auto dispatch_rnn_relu = [](const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
      };
      return wrap(dispatch_rnn_relu(_r.tensor(0), _r.tensor(1), _r.tensorlist(2), _r.toBool(3), _r.toInt64(4), _r.toDouble(5), _r.toBool(6), _r.toBool(7), _r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rnn_relu_cell
static PyObject * THPVariable_rnn_relu_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
  auto dispatch_rnn_relu_cell = [](const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  };
  return wrap(dispatch_rnn_relu_cell(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// rnn_tanh
static PyObject * THPVariable_rnn_tanh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rnn_tanh(Tensor data, Tensor batch_sizes, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)",
    "rnn_tanh(Tensor input, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::rnn_tanh.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
      auto dispatch_rnn_tanh = [](const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::rnn_tanh(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
      };
      return wrap(dispatch_rnn_tanh(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensorlist(3), _r.toBool(4), _r.toInt64(5), _r.toDouble(6), _r.toBool(7), _r.toBool(8)));
    }
    case 1: {
      // aten::rnn_tanh.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
      auto dispatch_rnn_tanh = [](const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::rnn_tanh(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
      };
      return wrap(dispatch_rnn_tanh(_r.tensor(0), _r.tensor(1), _r.tensorlist(2), _r.toBool(3), _r.toInt64(4), _r.toDouble(5), _r.toBool(6), _r.toBool(7), _r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rnn_tanh_cell
static PyObject * THPVariable_rnn_tanh_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
  auto dispatch_rnn_tanh_cell = [](const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  };
  return wrap(dispatch_rnn_tanh_cell(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// roll
static PyObject * THPVariable_roll(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "roll(Tensor input, IntArrayRef[1] shifts, IntArrayRef[1] dims=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor
  auto dispatch_roll = [](const Tensor & self, IntArrayRef shifts, IntArrayRef dims) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.roll(shifts, dims);
  };
  return wrap(dispatch_roll(_r.tensor(0), _r.intlist(1), _r.intlist(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rot90
static PyObject * THPVariable_rot90(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rot90(Tensor input, int64_t k=1, IntArrayRef dims={0,1})",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor
  auto dispatch_rot90 = [](const Tensor & self, int64_t k, IntArrayRef dims) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rot90(k, dims);
  };
  return wrap(dispatch_rot90(_r.tensor(0), _r.toInt64(1), _r.intlist(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// round
static PyObject * THPVariable_round(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "round(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::round(Tensor self) -> Tensor
    auto dispatch_round = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.round();
    };
    return wrap(dispatch_round(_r.tensor(0)));
  } else {
    // aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_round_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::round_out(out, self);
    };
    return wrap(dispatch_round_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// round_
static PyObject * THPVariable_round_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "round_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::round_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_round_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.round_();
  };
  return wrap(dispatch_round_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rrelu
static PyObject * THPVariable_rrelu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu(Tensor input, Scalar lower=0.125, Scalar upper=0.333333333333, bool training=False, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
  auto dispatch_rrelu = [](const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::rrelu(self, lower, upper, training, generator);
  };
  return wrap(dispatch_rrelu(_r.tensor(0), _r.scalar(1), _r.scalar(2), _r.toBool(3), _r.generator(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rrelu_
static PyObject * THPVariable_rrelu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu_(Tensor input, Scalar lower=0.125, Scalar upper=0.333333333333, bool training=False, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::rrelu_(Tensor(a!) self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)
  auto dispatch_rrelu_ = [](Tensor self, Scalar lower, Scalar upper, bool training, Generator * generator) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::rrelu_(self, lower, upper, training, generator);
  };
  return wrap(dispatch_rrelu_(_r.tensor(0), _r.scalar(1), _r.scalar(2), _r.toBool(3), _r.generator(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rsqrt
static PyObject * THPVariable_rsqrt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rsqrt(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::rsqrt(Tensor self) -> Tensor
    auto dispatch_rsqrt = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.rsqrt();
    };
    return wrap(dispatch_rsqrt(_r.tensor(0)));
  } else {
    // aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_rsqrt_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::rsqrt_out(out, self);
    };
    return wrap(dispatch_rsqrt_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rsqrt_
static PyObject * THPVariable_rsqrt_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rsqrt_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_rsqrt_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rsqrt_();
  };
  return wrap(dispatch_rsqrt_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// rsub
static PyObject * THPVariable_rsub(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rsub(Tensor input, Tensor other, *, Scalar alpha=1)",
    "rsub(Tensor input, Scalar other, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      auto dispatch_rsub = [](const Tensor & self, const Tensor & other, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::rsub(self, other, alpha);
      };
      return wrap(dispatch_rsub(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
    }
    case 1: {
      // aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
      auto dispatch_rsub = [](const Tensor & self, Scalar other, Scalar alpha) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::rsub(self, other, alpha);
      };
      return wrap(dispatch_rsub(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// scalar_tensor
static PyObject * THPVariable_scalar_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "scalar_tensor(Scalar s, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartype(1))
      .device(_r.device(3))
      .layout(_r.layout(2).layout)
      .requires_grad(_r.toBool(5))
      .pinned_memory(_r.toBool(4));
  torch::utils::maybe_initialize_cuda(options);
  auto dispatch_scalar_tensor = [](Scalar s, const TensorOptions & options) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::scalar_tensor(s, options);
  };
  return wrap(dispatch_scalar_tensor(_r.scalar(0), options));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// scatter
static PyObject * THPVariable_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "scatter(Tensor input, Dimname dim, Tensor index, Tensor src)",
    "scatter(Tensor input, int64_t dim, Tensor index, Tensor src)",
    "scatter(Tensor input, Dimname dim, Tensor index, Scalar value)",
    "scatter(Tensor input, int64_t dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::scatter.dimname_src(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
      auto dispatch_scatter = [](const Tensor & self, Dimname dim, const Tensor & index, const Tensor & src) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, src);
      };
      return wrap(dispatch_scatter(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
      auto dispatch_scatter = [](const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, src);
      };
      return wrap(dispatch_scatter(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3)));
    }
    case 2: {
      // aten::scatter.dimname_value(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
      auto dispatch_scatter = [](const Tensor & self, Dimname dim, const Tensor & index, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, value);
      };
      return wrap(dispatch_scatter(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.scalar(3)));
    }
    case 3: {
      // aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
      auto dispatch_scatter = [](const Tensor & self, int64_t dim, const Tensor & index, Scalar value) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, value);
      };
      return wrap(dispatch_scatter(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.scalar(3)));
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
  static PythonArgParser parser({
    "scatter_add(Tensor input, Dimname dim, Tensor index, Tensor src)",
    "scatter_add(Tensor input, int64_t dim, Tensor index, Tensor src)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::scatter_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
      auto dispatch_scatter_add = [](const Tensor & self, Dimname dim, const Tensor & index, const Tensor & src) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter_add(dim, index, src);
      };
      return wrap(dispatch_scatter_add(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
      auto dispatch_scatter_add = [](const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter_add(dim, index, src);
      };
      return wrap(dispatch_scatter_add(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// select
static PyObject * THPVariable_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "select(Tensor input, Dimname dim, int64_t index)",
    "select(Tensor input, int64_t dim, int64_t index)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)
      auto dispatch_select = [](const Tensor & self, Dimname dim, int64_t index) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.select(dim, index);
      };
      return wrap(dispatch_select(_r.tensor(0), _r.dimname(1), _r.toInt64(2)));
    }
    case 1: {
      // aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
      auto dispatch_select = [](const Tensor & self, int64_t dim, int64_t index) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.select(dim, index);
      };
      return wrap(dispatch_select(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// selu
static PyObject * THPVariable_selu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "selu(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::selu(Tensor self) -> Tensor
  auto dispatch_selu = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::selu(self);
  };
  return wrap(dispatch_selu(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// selu_
static PyObject * THPVariable_selu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "selu_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::selu_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_selu_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::selu_(self);
  };
  return wrap(dispatch_selu_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sigmoid
static PyObject * THPVariable_sigmoid(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sigmoid(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::sigmoid(Tensor self) -> Tensor
    auto dispatch_sigmoid = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.sigmoid();
    };
    return wrap(dispatch_sigmoid(_r.tensor(0)));
  } else {
    // aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_sigmoid_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::sigmoid_out(out, self);
    };
    return wrap(dispatch_sigmoid_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sigmoid_
static PyObject * THPVariable_sigmoid_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sigmoid_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_sigmoid_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sigmoid_();
  };
  return wrap(dispatch_sigmoid_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sign
static PyObject * THPVariable_sign(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sign(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::sign(Tensor self) -> Tensor
    auto dispatch_sign = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.sign();
    };
    return wrap(dispatch_sign(_r.tensor(0)));
  } else {
    // aten::sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_sign_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::sign_out(out, self);
    };
    return wrap(dispatch_sign_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sin
static PyObject * THPVariable_sin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sin(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::sin(Tensor self) -> Tensor
    auto dispatch_sin = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.sin();
    };
    return wrap(dispatch_sin(_r.tensor(0)));
  } else {
    // aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_sin_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::sin_out(out, self);
    };
    return wrap(dispatch_sin_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sin_
static PyObject * THPVariable_sin_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sin_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::sin_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_sin_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sin_();
  };
  return wrap(dispatch_sin_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sinh
static PyObject * THPVariable_sinh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sinh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::sinh(Tensor self) -> Tensor
    auto dispatch_sinh = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.sinh();
    };
    return wrap(dispatch_sinh(_r.tensor(0)));
  } else {
    // aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_sinh_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::sinh_out(out, self);
    };
    return wrap(dispatch_sinh_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sinh_
static PyObject * THPVariable_sinh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sinh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::sinh_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_sinh_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sinh_();
  };
  return wrap(dispatch_sinh_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slogdet
static PyObject * THPVariable_slogdet(PyObject* self_, PyObject* args, PyObject* kwargs)
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
  static PythonArgParser parser({
    "slogdet(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
  auto dispatch_slogdet = [](const Tensor & self) -> std::tuple<Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.slogdet();
  };
  return wrap(&NamedTuple, dispatch_slogdet(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// smm
static PyObject * THPVariable_smm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "smm(Tensor input, Tensor mat2)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::smm(Tensor self, Tensor mat2) -> Tensor
  auto dispatch_smm = [](const Tensor & self, const Tensor & mat2) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.smm(mat2);
  };
  return wrap(dispatch_smm(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// softmax
static PyObject * THPVariable_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "softmax(Tensor input, Dimname dim, *, ScalarType? dtype=None)",
    "softmax(Tensor input, int64_t dim, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_softmax = [](const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.softmax(dim, dtype);
      };
      return wrap(dispatch_softmax(_r.tensor(0), _r.dimname(1), _r.scalartypeOptional(2)));
    }
    case 1: {
      // aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
      auto dispatch_softmax = [](const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.softmax(dim, dtype);
      };
      return wrap(dispatch_softmax(_r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
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
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.solve_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "solve(Tensor input, Tensor A, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::solve(Tensor self, Tensor A) -> (Tensor solution, Tensor LU)
    auto dispatch_solve = [](const Tensor & self, const Tensor & A) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.solve(A);
    };
    return wrap(&NamedTuple, dispatch_solve(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::solve.solution(Tensor self, Tensor A, *, Tensor(a!) solution, Tensor(b!) lu) -> (Tensor(a!) solution, Tensor(b!) LU)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_solve_out = [](Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::solve_out(solution, lu, self, A);
    };
    return wrap(&NamedTuple1, dispatch_solve_out(out[0], out[1], _r.tensor(0), _r.tensor(1)));
  }
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
    static PyStructSequence_Desc desc = { "torch.return_types.sort_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.sort", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "sort(Tensor input, Dimname dim, bool descending=False, *, TensorList[2] out=None)",
    "sort(Tensor input, int64_t dim=-1, bool descending=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::sort.dimname(Tensor self, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)
        auto dispatch_sort = [](const Tensor & self, Dimname dim, bool descending) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.sort(dim, descending);
        };
        return wrap(&NamedTuple1, dispatch_sort(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::sort.dimname_values(Tensor self, Dimname dim, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_sort_out = [](Tensor & values, Tensor & indices, const Tensor & self, Dimname dim, bool descending) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::sort_out(values, indices, self, dim, descending);
        };
        return wrap(&NamedTuple, dispatch_sort_out(out[0], out[1], _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
        auto dispatch_sort = [](const Tensor & self, int64_t dim, bool descending) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.sort(dim, descending);
        };
        return wrap(&NamedTuple1, dispatch_sort(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::sort.values(Tensor self, int dim=-1, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_sort_out = [](Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) -> std::tuple<Tensor,Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::sort_out(values, indices, self, dim, descending);
        };
        return wrap(&NamedTuple, dispatch_sort_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// split
static PyObject * THPVariable_split(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "split(Tensor input, int64_t split_size, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]
  auto dispatch_split = [](const Tensor & self, int64_t split_size, int64_t dim) -> std::vector<Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.split(split_size, dim);
  };
  return wrap(dispatch_split(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// split_with_sizes
static PyObject * THPVariable_split_with_sizes(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "split_with_sizes(Tensor input, IntArrayRef split_sizes, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]
  auto dispatch_split_with_sizes = [](const Tensor & self, IntArrayRef split_sizes, int64_t dim) -> std::vector<Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.split_with_sizes(split_sizes, dim);
  };
  return wrap(dispatch_split_with_sizes(_r.tensor(0), _r.intlist(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sqrt
static PyObject * THPVariable_sqrt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sqrt(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::sqrt(Tensor self) -> Tensor
    auto dispatch_sqrt = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.sqrt();
    };
    return wrap(dispatch_sqrt(_r.tensor(0)));
  } else {
    // aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_sqrt_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::sqrt_out(out, self);
    };
    return wrap(dispatch_sqrt_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sqrt_
static PyObject * THPVariable_sqrt_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sqrt_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::sqrt_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_sqrt_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sqrt_();
  };
  return wrap(dispatch_sqrt_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// square
static PyObject * THPVariable_square(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "square(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::square(Tensor self) -> Tensor
  auto dispatch_square = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.square();
  };
  return wrap(dispatch_square(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// square_
static PyObject * THPVariable_square_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "square_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::square_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_square_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.square_();
  };
  return wrap(dispatch_square_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// squeeze
static PyObject * THPVariable_squeeze(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "squeeze(Tensor input)",
    "squeeze(Tensor input, Dimname dim)",
    "squeeze(Tensor input, int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::squeeze(Tensor(a) self) -> Tensor(a)
      auto dispatch_squeeze = [](const Tensor & self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze();
      };
      return wrap(dispatch_squeeze(_r.tensor(0)));
    }
    case 1: {
      // aten::squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)
      auto dispatch_squeeze = [](const Tensor & self, Dimname dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze(dim);
      };
      return wrap(dispatch_squeeze(_r.tensor(0), _r.dimname(1)));
    }
    case 2: {
      // aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
      auto dispatch_squeeze = [](const Tensor & self, int64_t dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze(dim);
      };
      return wrap(dispatch_squeeze(_r.tensor(0), _r.toInt64(1)));
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
  static PythonArgParser parser({
    "sspaddmm(Scalar beta, Tensor input, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Scalar beta, Tensor input, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Tensor input, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_sspaddmm = [](Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sspaddmm(mat1, mat2, beta, alpha);
      };
      return wrap(dispatch_sspaddmm(_r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
    }
    case 1: {
      // [deprecated] aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      auto dispatch_sspaddmm = [](Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sspaddmm(mat1, mat2, beta, 1);
      };
      return wrap(dispatch_sspaddmm(_r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
    }
    case 2: {
      if (_r.isNone(5)) {
        // aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        auto dispatch_sspaddmm = [](const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.sspaddmm(mat1, mat2, beta, alpha);
        };
        return wrap(dispatch_sspaddmm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      } else {
        // aten::sspaddmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_sspaddmm_out = [](Tensor out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::sspaddmm_out(out, self, mat1, mat2, beta, alpha);
        };
        return wrap(dispatch_sspaddmm_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// stack
static PyObject * THPVariable_stack(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "stack(TensorList tensors, int64_t dim=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::stack(Tensor[] tensors, int dim=0) -> Tensor
    auto dispatch_stack = [](TensorList tensors, int64_t dim) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::stack(tensors, dim);
    };
    return wrap(dispatch_stack(_r.tensorlist(0), _r.toInt64(1)));
  } else {
    // aten::stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_stack_out = [](Tensor out, TensorList tensors, int64_t dim) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::stack_out(out, tensors, dim);
    };
    return wrap(dispatch_stack_out(_r.tensor(2), _r.tensorlist(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// std
static PyObject * THPVariable_std(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "std(Tensor input, DimnameList[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor out=None)",
    "std(Tensor input, IntArrayRef[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor out=None)",
    "std(Tensor input, bool unbiased=True)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::std.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
        auto dispatch_std = [](const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.std(dim, unbiased, keepdim);
        };
        return wrap(dispatch_std(_r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.toBool(3)));
      } else {
        // aten::std.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_std_out = [](Tensor out, const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::std_out(out, self, dim, unbiased, keepdim);
        };
        return wrap(dispatch_std_out(_r.tensor(4), _r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.toBool(3)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::std.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
        auto dispatch_std = [](const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.std(dim, unbiased, keepdim);
        };
        return wrap(dispatch_std(_r.tensor(0), _r.intlist(1), _r.toBool(2), _r.toBool(3)));
      } else {
        // aten::std.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_std_out = [](Tensor out, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::std_out(out, self, dim, unbiased, keepdim);
        };
        return wrap(dispatch_std_out(_r.tensor(4), _r.tensor(0), _r.intlist(1), _r.toBool(2), _r.toBool(3)));
      }
    }
    case 2: {
      // aten::std(Tensor self, bool unbiased=True) -> Tensor
      auto dispatch_std = [](const Tensor & self, bool unbiased) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.std(unbiased);
      };
      return wrap(dispatch_std(_r.tensor(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// std_mean
static PyObject * THPVariable_std_mean(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "std_mean(Tensor input, DimnameList[1] dim, bool unbiased=True, bool keepdim=False)",
    "std_mean(Tensor input, IntArrayRef[1] dim, bool unbiased=True, bool keepdim=False)",
    "std_mean(Tensor input, bool unbiased=True)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::std_mean.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
      auto dispatch_std_mean = [](const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::std_mean(self, dim, unbiased, keepdim);
      };
      return wrap(dispatch_std_mean(_r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.toBool(3)));
    }
    case 1: {
      // aten::std_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
      auto dispatch_std_mean = [](const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::std_mean(self, dim, unbiased, keepdim);
      };
      return wrap(dispatch_std_mean(_r.tensor(0), _r.intlist(1), _r.toBool(2), _r.toBool(3)));
    }
    case 2: {
      // aten::std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
      auto dispatch_std_mean = [](const Tensor & self, bool unbiased) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::std_mean(self, unbiased);
      };
      return wrap(dispatch_std_mean(_r.tensor(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// stft
static PyObject * THPVariable_stft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "stft(Tensor input, int64_t n_fft, int64_t? hop_length=None, int64_t? win_length=None, Tensor? window=None, bool normalized=False, bool onesided=True)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool onesided=True) -> Tensor
  auto dispatch_stft = [](const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.stft(n_fft, hop_length, win_length, window, normalized, onesided);
  };
  return wrap(dispatch_stft(_r.tensor(0), _r.toInt64(1), _r.toInt64Optional(2), _r.toInt64Optional(3), _r.tensor(4), _r.toBool(5), _r.toBool(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// sub
static PyObject * THPVariable_sub(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sub(Tensor input, Scalar alpha, Tensor other, *, Tensor out=None)|deprecated",
    "sub(Tensor input, Tensor other, *, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // [deprecated] aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
        auto dispatch_sub = [](const Tensor & self, Scalar alpha, const Tensor & other) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.sub(other, alpha);
        };
        return wrap(dispatch_sub(_r.tensor(0), _r.scalar(1), _r.tensor(2)));
      } else {
        // [deprecated] aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_sub_out = [](const Tensor & self, Scalar alpha, const Tensor & other, Tensor out) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::sub_out(out, self, other, alpha);
        };
        return wrap(dispatch_sub_out(_r.tensor(0), _r.scalar(1), _r.tensor(2), _r.tensor(3)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
        auto dispatch_sub = [](const Tensor & self, const Tensor & other, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.sub(other, alpha);
        };
        return wrap(dispatch_sub(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
      } else {
        // aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_sub_out = [](Tensor out, const Tensor & self, const Tensor & other, Scalar alpha) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::sub_out(out, self, other, alpha);
        };
        return wrap(dispatch_sub_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.scalar(2)));
      }
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
  static PythonArgParser parser({
    "sum(Tensor input, *, ScalarType? dtype=None)",
    "sum(Tensor input, DimnameList[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
    "sum(Tensor input, IntArrayRef[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
      auto dispatch_sum = [](const Tensor & self, c10::optional<ScalarType> dtype) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sum(dtype);
      };
      return wrap(dispatch_sum(_r.tensor(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        auto dispatch_sum = [](const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.sum(dim, keepdim, dtype);
        };
        return wrap(dispatch_sum(_r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.scalartypeOptional(3)));
      } else {
        // aten::sum.DimnameList_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_sum_out = [](Tensor out, const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::sum_out(out, self, dim, keepdim, dtype);
        };
        return wrap(dispatch_sum_out(_r.tensor(4), _r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.scalartypeOptional(3)));
      }
    }
    case 2: {
      if (_r.isNone(4)) {
        // aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        auto dispatch_sum = [](const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.sum(dim, keepdim, dtype);
        };
        return wrap(dispatch_sum(_r.tensor(0), _r.intlist(1), _r.toBool(2), _r.scalartypeOptional(3)));
      } else {
        // aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_sum_out = [](Tensor out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::sum_out(out, self, dim, keepdim, dtype);
        };
        return wrap(dispatch_sum_out(_r.tensor(4), _r.tensor(0), _r.intlist(1), _r.toBool(2), _r.scalartypeOptional(3)));
      }
    }
  }
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
    static PyStructSequence_Desc desc = { "torch.return_types.svd_out", nullptr, NamedTuple_fields, 3 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.svd", nullptr, NamedTuple_fields, 3 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "svd(Tensor input, bool some=True, bool compute_uv=True, *, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(3)) {
    // aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)
    auto dispatch_svd = [](const Tensor & self, bool some, bool compute_uv) -> std::tuple<Tensor,Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.svd(some, compute_uv);
    };
    return wrap(&NamedTuple1, dispatch_svd(_r.tensor(0), _r.toBool(1), _r.toBool(2)));
  } else {
    // aten::svd.U(Tensor self, bool some=True, bool compute_uv=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) V) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) V)
    auto out = _r.tensorlist_n<3>(3);
    auto dispatch_svd_out = [](Tensor & U, Tensor & S, Tensor & V, const Tensor & self, bool some, bool compute_uv) -> std::tuple<Tensor,Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::svd_out(U, S, V, self, some, compute_uv);
    };
    return wrap(&NamedTuple, dispatch_svd_out(out[0], out[1], out[2], _r.tensor(0), _r.toBool(1), _r.toBool(2)));
  }
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
    static PyStructSequence_Desc desc = { "torch.return_types.symeig_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.symeig", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "symeig(Tensor input, bool eigenvectors=False, bool upper=True, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(3)) {
    // aten::symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor eigenvalues, Tensor eigenvectors)
    auto dispatch_symeig = [](const Tensor & self, bool eigenvectors, bool upper) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.symeig(eigenvectors, upper);
    };
    return wrap(&NamedTuple1, dispatch_symeig(_r.tensor(0), _r.toBool(1), _r.toBool(2)));
  } else {
    // aten::symeig.e(Tensor self, bool eigenvectors=False, bool upper=True, *, Tensor(a!) e, Tensor(b!) V) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
    auto out = _r.tensorlist_n<2>(3);
    auto dispatch_symeig_out = [](Tensor & e, Tensor & V, const Tensor & self, bool eigenvectors, bool upper) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::symeig_out(e, V, self, eigenvectors, upper);
    };
    return wrap(&NamedTuple, dispatch_symeig_out(out[0], out[1], _r.tensor(0), _r.toBool(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// t
static PyObject * THPVariable_t(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "t(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::t(Tensor(a) self) -> Tensor(a)
  auto dispatch_t = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.t();
  };
  return wrap(dispatch_t(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// take
static PyObject * THPVariable_take(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "take(Tensor input, Tensor index, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::take(Tensor self, Tensor index) -> Tensor
    auto dispatch_take = [](const Tensor & self, const Tensor & index) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.take(index);
    };
    return wrap(dispatch_take(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_take_out = [](Tensor out, const Tensor & self, const Tensor & index) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::take_out(out, self, index);
    };
    return wrap(dispatch_take_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tan
static PyObject * THPVariable_tan(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tan(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::tan(Tensor self) -> Tensor
    auto dispatch_tan = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.tan();
    };
    return wrap(dispatch_tan(_r.tensor(0)));
  } else {
    // aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_tan_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::tan_out(out, self);
    };
    return wrap(dispatch_tan_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tan_
static PyObject * THPVariable_tan_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tan_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::tan_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_tan_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tan_();
  };
  return wrap(dispatch_tan_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tanh
static PyObject * THPVariable_tanh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tanh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::tanh(Tensor self) -> Tensor
    auto dispatch_tanh = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.tanh();
    };
    return wrap(dispatch_tanh(_r.tensor(0)));
  } else {
    // aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_tanh_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::tanh_out(out, self);
    };
    return wrap(dispatch_tanh_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tanh_
static PyObject * THPVariable_tanh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tanh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::tanh_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_tanh_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tanh_();
  };
  return wrap(dispatch_tanh_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tensordot
static PyObject * THPVariable_tensordot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tensordot(Tensor input, Tensor other, IntArrayRef dims_self, IntArrayRef dims_other)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor
  auto dispatch_tensordot = [](const Tensor & self, const Tensor & other, IntArrayRef dims_self, IntArrayRef dims_other) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::tensordot(self, other, dims_self, dims_other);
  };
  return wrap(dispatch_tensordot(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// threshold
static PyObject * THPVariable_threshold(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "threshold(Tensor input, Scalar threshold, Scalar value, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(3)) {
    // aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor
    auto dispatch_threshold = [](const Tensor & self, Scalar threshold, Scalar value) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::threshold(self, threshold, value);
    };
    return wrap(dispatch_threshold(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
  } else {
    // aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_threshold_out = [](Tensor out, const Tensor & self, Scalar threshold, Scalar value) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::threshold_out(out, self, threshold, value);
    };
    return wrap(dispatch_threshold_out(_r.tensor(3), _r.tensor(0), _r.scalar(1), _r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// threshold_
static PyObject * THPVariable_threshold_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "threshold_(Tensor input, Scalar threshold, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)
  auto dispatch_threshold_ = [](Tensor self, Scalar threshold, Scalar value) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::threshold_(self, threshold, value);
  };
  return wrap(dispatch_threshold_(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
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
    static PyStructSequence_Desc desc = { "torch.return_types.topk_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.topk", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "topk(Tensor input, int64_t k, int64_t dim=-1, bool largest=True, bool sorted=True, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(5)) {
    // aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
    auto dispatch_topk = [](const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.topk(k, dim, largest, sorted);
    };
    return wrap(&NamedTuple1, dispatch_topk(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toBool(3), _r.toBool(4)));
  } else {
    // aten::topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) ->(Tensor(a!) values, Tensor(b!) indices)
    auto out = _r.tensorlist_n<2>(5);
    auto dispatch_topk_out = [](Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::topk_out(values, indices, self, k, dim, largest, sorted);
    };
    return wrap(&NamedTuple, dispatch_topk_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toBool(3), _r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// trace
static PyObject * THPVariable_trace(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trace(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::trace(Tensor self) -> Tensor
  auto dispatch_trace = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.trace();
  };
  return wrap(dispatch_trace(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// transpose
static PyObject * THPVariable_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "transpose(Tensor input, Dimname dim0, Dimname dim1)",
    "transpose(Tensor input, int64_t dim0, int64_t dim1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)
      auto dispatch_transpose = [](const Tensor & self, Dimname dim0, Dimname dim1) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.transpose(dim0, dim1);
      };
      return wrap(dispatch_transpose(_r.tensor(0), _r.dimname(1), _r.dimname(2)));
    }
    case 1: {
      // aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
      auto dispatch_transpose = [](const Tensor & self, int64_t dim0, int64_t dim1) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.transpose(dim0, dim1);
      };
      return wrap(dispatch_transpose(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// trapz
static PyObject * THPVariable_trapz(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trapz(Tensor y, *, double dx=1, int64_t dim=-1)",
    "trapz(Tensor y, Tensor x, *, int64_t dim=-1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::trapz.dx(Tensor y, *, float dx=1, int dim=-1) -> Tensor
      auto dispatch_trapz = [](const Tensor & y, double dx, int64_t dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::trapz(y, dx, dim);
      };
      return wrap(dispatch_trapz(_r.tensor(0), _r.toDouble(1), _r.toInt64(2)));
    }
    case 1: {
      // aten::trapz.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
      auto dispatch_trapz = [](const Tensor & y, const Tensor & x, int64_t dim) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::trapz(y, x, dim);
      };
      return wrap(dispatch_trapz(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
    }
  }
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
    static PyStructSequence_Desc desc = { "torch.return_types.triangular_solve_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.triangular_solve", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "triangular_solve(Tensor input, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(5)) {
    // aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)
    auto dispatch_triangular_solve = [](const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.triangular_solve(A, upper, transpose, unitriangular);
    };
    return wrap(&NamedTuple1, dispatch_triangular_solve(_r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3), _r.toBool(4)));
  } else {
    // aten::triangular_solve.X(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, Tensor(a!) X, Tensor(b!) M) -> (Tensor(a!) solution, Tensor(b!) cloned_coefficient)
    auto out = _r.tensorlist_n<2>(5);
    auto dispatch_triangular_solve_out = [](Tensor & X, Tensor & M, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::triangular_solve_out(X, M, self, A, upper, transpose, unitriangular);
    };
    return wrap(&NamedTuple, dispatch_triangular_solve_out(out[0], out[1], _r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3), _r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tril
static PyObject * THPVariable_tril(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tril(Tensor input, int64_t diagonal=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::tril(Tensor self, int diagonal=0) -> Tensor
    auto dispatch_tril = [](const Tensor & self, int64_t diagonal) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.tril(diagonal);
    };
    return wrap(dispatch_tril(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_tril_out = [](Tensor out, const Tensor & self, int64_t diagonal) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::tril_out(out, self, diagonal);
    };
    return wrap(dispatch_tril_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tril_indices
static PyObject * THPVariable_tril_indices(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tril_indices(int64_t row, int64_t col, int64_t offset=0, *, ScalarType dtype=torch.int64, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartype(3))
      .device(_r.device(5))
      .layout(_r.layout(4).layout)
      .requires_grad(_r.toBool(7))
      .pinned_memory(_r.toBool(6));
  torch::utils::maybe_initialize_cuda(options);
  auto dispatch_tril_indices = [](int64_t row, int64_t col, int64_t offset, const TensorOptions & options) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::tril_indices(row, col, offset, options);
  };
  return wrap(dispatch_tril_indices(_r.toInt64(0), _r.toInt64(1), _r.toInt64(2), options));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// triplet_margin_loss
static PyObject * THPVariable_triplet_margin_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, double margin=1.0, double p=2, double eps=1e-06, bool swap=False, int64_t reduction=at::Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin=1.0, float p=2, float eps=1e-06, bool swap=False, int reduction=Mean) -> Tensor
  auto dispatch_triplet_margin_loss = [](const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, reduction);
  };
  return wrap(dispatch_triplet_margin_loss(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toDouble(3), _r.toDouble(4), _r.toDouble(5), _r.toBool(6), _r.toInt64(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// triu
static PyObject * THPVariable_triu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triu(Tensor input, int64_t diagonal=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(2)) {
    // aten::triu(Tensor self, int diagonal=0) -> Tensor
    auto dispatch_triu = [](const Tensor & self, int64_t diagonal) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.triu(diagonal);
    };
    return wrap(dispatch_triu(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::triu.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_triu_out = [](Tensor out, const Tensor & self, int64_t diagonal) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::triu_out(out, self, diagonal);
    };
    return wrap(dispatch_triu_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// triu_indices
static PyObject * THPVariable_triu_indices(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triu_indices(int64_t row, int64_t col, int64_t offset=0, *, ScalarType dtype=torch.int64, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartype(3))
      .device(_r.device(5))
      .layout(_r.layout(4).layout)
      .requires_grad(_r.toBool(7))
      .pinned_memory(_r.toBool(6));
  torch::utils::maybe_initialize_cuda(options);
  auto dispatch_triu_indices = [](int64_t row, int64_t col, int64_t offset, const TensorOptions & options) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::triu_indices(row, col, offset, options);
  };
  return wrap(dispatch_triu_indices(_r.toInt64(0), _r.toInt64(1), _r.toInt64(2), options));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// trunc
static PyObject * THPVariable_trunc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trunc(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  if (_r.isNone(1)) {
    // aten::trunc(Tensor self) -> Tensor
    auto dispatch_trunc = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.trunc();
    };
    return wrap(dispatch_trunc(_r.tensor(0)));
  } else {
    // aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_trunc_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::trunc_out(out, self);
    };
    return wrap(dispatch_trunc_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// trunc_
static PyObject * THPVariable_trunc_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trunc_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::trunc_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_trunc_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.trunc_();
  };
  return wrap(dispatch_trunc_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// unbind
static PyObject * THPVariable_unbind(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unbind(Tensor input, Dimname dim)",
    "unbind(Tensor input, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::unbind.Dimname(Tensor(a) self, Dimname dim) -> Tensor(a)[]
      auto dispatch_unbind = [](const Tensor & self, Dimname dim) -> std::vector<Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.unbind(dim);
      };
      return wrap(dispatch_unbind(_r.tensor(0), _r.dimname(1)));
    }
    case 1: {
      // aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]
      auto dispatch_unbind = [](const Tensor & self, int64_t dim) -> std::vector<Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.unbind(dim);
      };
      return wrap(dispatch_unbind(_r.tensor(0), _r.toInt64(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unique_consecutive
static PyObject * THPVariable_unique_consecutive(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unique_consecutive(Tensor input, bool return_inverse=False, bool return_counts=False, int64_t? dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)
  auto dispatch_unique_consecutive = [](const Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim) -> std::tuple<Tensor,Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::unique_consecutive(self, return_inverse, return_counts, dim);
  };
  return wrap(dispatch_unique_consecutive(_r.tensor(0), _r.toBool(1), _r.toBool(2), _r.toInt64Optional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unique_dim
static PyObject * THPVariable_unique_dim(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unique_dim(Tensor input, int64_t dim, bool sorted=True, bool return_inverse=False, bool return_counts=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
  auto dispatch_unique_dim = [](const Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) -> std::tuple<Tensor,Tensor,Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::unique_dim(self, dim, sorted, return_inverse, return_counts);
  };
  return wrap(dispatch_unique_dim(_r.tensor(0), _r.toInt64(1), _r.toBool(2), _r.toBool(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unsqueeze
static PyObject * THPVariable_unsqueeze(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unsqueeze(Tensor input, int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
  auto dispatch_unsqueeze = [](const Tensor & self, int64_t dim) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.unsqueeze(dim);
  };
  return wrap(dispatch_unsqueeze(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// var
static PyObject * THPVariable_var(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "var(Tensor input, DimnameList[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor out=None)",
    "var(Tensor input, IntArrayRef[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor out=None)",
    "var(Tensor input, bool unbiased=True)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::var.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
        auto dispatch_var = [](const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.var(dim, unbiased, keepdim);
        };
        return wrap(dispatch_var(_r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.toBool(3)));
      } else {
        // aten::var.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_var_out = [](Tensor out, const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::var_out(out, self, dim, unbiased, keepdim);
        };
        return wrap(dispatch_var_out(_r.tensor(4), _r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.toBool(3)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::var.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
        auto dispatch_var = [](const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.var(dim, unbiased, keepdim);
        };
        return wrap(dispatch_var(_r.tensor(0), _r.intlist(1), _r.toBool(2), _r.toBool(3)));
      } else {
        // aten::var.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        auto dispatch_var_out = [](Tensor out, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::var_out(out, self, dim, unbiased, keepdim);
        };
        return wrap(dispatch_var_out(_r.tensor(4), _r.tensor(0), _r.intlist(1), _r.toBool(2), _r.toBool(3)));
      }
    }
    case 2: {
      // aten::var(Tensor self, bool unbiased=True) -> Tensor
      auto dispatch_var = [](const Tensor & self, bool unbiased) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.var(unbiased);
      };
      return wrap(dispatch_var(_r.tensor(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// var_mean
static PyObject * THPVariable_var_mean(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "var_mean(Tensor input, DimnameList[1] dim, bool unbiased=True, bool keepdim=False)",
    "var_mean(Tensor input, IntArrayRef[1] dim, bool unbiased=True, bool keepdim=False)",
    "var_mean(Tensor input, bool unbiased=True)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::var_mean.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
      auto dispatch_var_mean = [](const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::var_mean(self, dim, unbiased, keepdim);
      };
      return wrap(dispatch_var_mean(_r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.toBool(3)));
    }
    case 1: {
      // aten::var_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
      auto dispatch_var_mean = [](const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::var_mean(self, dim, unbiased, keepdim);
      };
      return wrap(dispatch_var_mean(_r.tensor(0), _r.intlist(1), _r.toBool(2), _r.toBool(3)));
    }
    case 2: {
      // aten::var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
      auto dispatch_var_mean = [](const Tensor & self, bool unbiased) -> std::tuple<Tensor,Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::var_mean(self, unbiased);
      };
      return wrap(dispatch_var_mean(_r.tensor(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// where
static PyObject * THPVariable_where(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "where(Tensor condition)",
    "where(Tensor condition, Tensor input, Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::where(Tensor condition) -> Tensor[]
      auto dispatch_where = [](const Tensor & condition) -> std::vector<Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::where(condition);
      };
      return wrap(dispatch_where(_r.tensor(0)));
    }
    case 1: {
      // aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
      auto dispatch_where = [](const Tensor & condition, const Tensor & self, const Tensor & other) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.where(condition, other);
      };
      return wrap(dispatch_where(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// zero_
static PyObject * THPVariable_zero_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "zero_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  // aten::zero_(Tensor(a!) self) -> Tensor(a!)
  auto dispatch_zero_ = [](Tensor self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.zero_();
  };
  return wrap(dispatch_zero_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// zeros
static PyObject * THPVariable_zeros(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "zeros(IntArrayRef size, *, DimnameList? names, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "zeros(IntArrayRef size, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::zeros.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(1);
      c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartype(2))
          .device(_r.device(4))
          .layout(_r.layout(3).layout)
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_zeros = [](IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::zeros(size, names, options);
      };
      return wrap(dispatch_zeros(_r.intlist(0), names, options));
    }
    case 1: {
      if (_r.isNone(1)) {
        // aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartype(2))
            .device(_r.device(4))
            .layout(_r.layout(3).layout)
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch::utils::maybe_initialize_cuda(options);
        auto dispatch_zeros = [](IntArrayRef size, const TensorOptions & options) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::zeros(size, options);
        };
        return wrap(dispatch_zeros(_r.intlist(0), options));
      } else {
        // aten::zeros.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartype(2), _r.isNone(2),
                               _r.layout(3), _r.isNone(3),
                               _r.device(4), _r.isNone(4));
        auto dispatch_zeros_out = [](Tensor out, IntArrayRef size) -> Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::zeros_out(out, size);
        };
        return wrap(dispatch_zeros_out(_r.tensor(1), _r.intlist(0)).set_requires_grad(_r.toBool(6)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// zeros_like
static PyObject * THPVariable_zeros_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "zeros_like(Tensor input, *, MemoryFormat? memory_format=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "zeros_like(Tensor input, *, MemoryFormat? memory_format=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  if (_r.has_torch_function()) {
    return handle_torch_function(_r, args, kwargs, THPVariableFunctions);
  }
  switch (_r.idx) {
    case 0: {
      // aten::zeros_like.dtype(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor
      auto self = _r.tensor(0);
      const auto options = TensorOptions()
          .dtype(_r.scalartypeWithDefault(2, self.scalar_type()))
          .device(_r.deviceWithDefault(4, self.device()))
          .layout(_r.layoutWithDefault(3, *torch::getLayout(self.options().backend())).layout)
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_cuda(options);
      auto dispatch_zeros_like = [](const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::zeros_like(self, options, memory_format);
      };
      return wrap(dispatch_zeros_like(self, options, _r.memoryformatOptional(1)));
    }
    case 1: {
      // aten::zeros_like(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
      auto dispatch_zeros_like = [](const Tensor & self, c10::optional<MemoryFormat> memory_format) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::zeros_like(self, memory_format);
      };
      return wrap(dispatch_zeros_like(_r.tensor(0), _r.memoryformatOptional(1)).set_requires_grad(_r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_nonzero(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nonzero(Tensor input, *, Tensor out=None)|deprecated",
    "nonzero(Tensor input, *, bool as_tuple)",
  });
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, args, kwargs, THPVariableFunctions);
  }

  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_nonzero(r.tensor(0)));
    } else {
      return wrap(dispatch_nonzero(r.tensor(0), r.tensor(1)));
    }
  } else {
    if (r.toBool(1)) {
      return wrap(dispatch_nonzero_numpy(r.tensor(0)));
    } else {
      return wrap(dispatch_nonzero(r.tensor(0)));
    }
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_numel(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "numel(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, args, kwargs, THPVariableFunctions);
  }

  if (r.idx == 0) {
    return wrap(r.tensor(0).numel());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
}} // namespace torch::autograd
