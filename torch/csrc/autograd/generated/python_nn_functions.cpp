// @generated from tools/autograd/templates/python_nn_functions.cpp

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_nn_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"

using at::Tensor;
using at::Scalar;
using at::MemoryFormat;
using at::Generator;
using at::IntArrayRef;

using namespace torch::autograd::utils;

namespace torch { namespace autograd {

static PyObject * THPVariable__parse_to(PyObject* module, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto parsed = parse_to_conversion(args, kwargs, /*allow_copy*/ false); // we don't want copy for nn.Module.to
  auto& device = std::get<0>(parsed);
  auto& scalarType = std::get<1>(parsed);
  auto non_blocking = std::get<2>(parsed);
  auto opt_memory_format = std::get<4>(parsed);
  auto tuple = THPObjectPtr{PyTuple_New(4)};
  if (!tuple) throw python_error();
  if (device) {
    PyTuple_SET_ITEM(tuple.get(), 0, THPDevice_New(*device));
  } else {
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(tuple.get(), 0, Py_None);
  }
  if (scalarType) {
    PyTuple_SET_ITEM(tuple.get(), 1, torch::autograd::utils::wrap(torch::getDtype(*scalarType)));
  } else {
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(tuple.get(), 1, Py_None);
  }
  PyTuple_SET_ITEM(tuple.get(), 2, torch::autograd::utils::wrap(non_blocking));
  if (opt_memory_format.has_value()) {
    PyTuple_SET_ITEM(tuple.get(), 3, THPMemoryFormat_New(opt_memory_format.value(), "unused_name"));
  } else {
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(tuple.get(), 3, Py_None);
  }
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

// adaptive_avg_pool2d
static PyObject * THPVariable_adaptive_avg_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_avg_pool2d(Tensor input, IntArrayRef[2] output_size, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(2)) {
    // aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
    auto dispatch_adaptive_avg_pool2d = [](const Tensor & self, IntArrayRef output_size) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_avg_pool2d(self, output_size);
    };
    return wrap(dispatch_adaptive_avg_pool2d(_r.tensor(0), _r.intlist(1)));
  } else {
    // aten::adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_adaptive_avg_pool2d_out = [](Tensor out, const Tensor & self, IntArrayRef output_size) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_avg_pool2d_out(out, self, output_size);
    };
    return wrap(dispatch_adaptive_avg_pool2d_out(_r.tensor(2), _r.tensor(0), _r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// adaptive_avg_pool3d
static PyObject * THPVariable_adaptive_avg_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_avg_pool3d(Tensor input, IntArrayRef[3] output_size, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(2)) {
    // aten::adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor
    auto dispatch_adaptive_avg_pool3d = [](const Tensor & self, IntArrayRef output_size) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_avg_pool3d(self, output_size);
    };
    return wrap(dispatch_adaptive_avg_pool3d(_r.tensor(0), _r.intlist(1)));
  } else {
    // aten::adaptive_avg_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_adaptive_avg_pool3d_out = [](Tensor out, const Tensor & self, IntArrayRef output_size) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_avg_pool3d_out(out, self, output_size);
    };
    return wrap(dispatch_adaptive_avg_pool3d_out(_r.tensor(2), _r.tensor(0), _r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// adaptive_max_pool2d
static PyObject * THPVariable_adaptive_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_max_pool2d(Tensor input, IntArrayRef[2] output_size, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(2)) {
    // aten::adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)
    auto dispatch_adaptive_max_pool2d = [](const Tensor & self, IntArrayRef output_size) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_max_pool2d(self, output_size);
    };
    return wrap(dispatch_adaptive_max_pool2d(_r.tensor(0), _r.intlist(1)));
  } else {
    // aten::adaptive_max_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_adaptive_max_pool2d_out = [](Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_max_pool2d_out(out, indices, self, output_size);
    };
    return wrap(dispatch_adaptive_max_pool2d_out(out[0], out[1], _r.tensor(0), _r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// adaptive_max_pool3d
static PyObject * THPVariable_adaptive_max_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_max_pool3d(Tensor input, IntArrayRef[3] output_size, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(2)) {
    // aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)
    auto dispatch_adaptive_max_pool3d = [](const Tensor & self, IntArrayRef output_size) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_max_pool3d(self, output_size);
    };
    return wrap(dispatch_adaptive_max_pool3d(_r.tensor(0), _r.intlist(1)));
  } else {
    // aten::adaptive_max_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_adaptive_max_pool3d_out = [](Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_max_pool3d_out(out, indices, self, output_size);
    };
    return wrap(dispatch_adaptive_max_pool3d_out(out[0], out[1], _r.tensor(0), _r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// avg_pool2d
static PyObject * THPVariable_avg_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "avg_pool2d(Tensor input, IntArrayRef[2] kernel_size, IntArrayRef[2] stride=None, IntArrayRef[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int64_t? divisor_override=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(7)) {
    // aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    auto dispatch_avg_pool2d = [](const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    };
    return wrap(dispatch_avg_pool2d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.toBool(4), _r.toBool(5), _r.toInt64Optional(6)));
  } else {
    // aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_avg_pool2d_out = [](Tensor out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::avg_pool2d_out(out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    };
    return wrap(dispatch_avg_pool2d_out(_r.tensor(7), _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.toBool(4), _r.toBool(5), _r.toInt64Optional(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// avg_pool3d
static PyObject * THPVariable_avg_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "avg_pool3d(Tensor input, IntArrayRef[3] kernel_size, IntArrayRef[3] stride=None, IntArrayRef[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int64_t? divisor_override=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(7)) {
    // aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    auto dispatch_avg_pool3d = [](const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    };
    return wrap(dispatch_avg_pool3d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.toBool(4), _r.toBool(5), _r.toInt64Optional(6)));
  } else {
    // aten::avg_pool3d.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_avg_pool3d_out = [](Tensor out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::avg_pool3d_out(out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    };
    return wrap(dispatch_avg_pool3d_out(_r.tensor(7), _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.toBool(4), _r.toBool(5), _r.toInt64Optional(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// binary_cross_entropy
static PyObject * THPVariable_binary_cross_entropy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "binary_cross_entropy(Tensor input, Tensor target, Tensor? weight=None, int64_t reduction=at::Reduction::Mean, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(4)) {
    // aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
    auto dispatch_binary_cross_entropy = [](const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::binary_cross_entropy(self, target, weight, reduction);
    };
    return wrap(dispatch_binary_cross_entropy(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3)));
  } else {
    // aten::binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_binary_cross_entropy_out = [](Tensor out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::binary_cross_entropy_out(out, self, target, weight, reduction);
    };
    return wrap(dispatch_binary_cross_entropy_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// col2im
static PyObject * THPVariable_col2im(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "col2im(Tensor input, IntArrayRef[2] output_size, IntArrayRef[2] kernel_size, IntArrayRef[2] dilation, IntArrayRef[2] padding, IntArrayRef[2] stride, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(6)) {
    // aten::col2im(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
    auto dispatch_col2im = [](const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::col2im(self, output_size, kernel_size, dilation, padding, stride);
    };
    return wrap(dispatch_col2im(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.intlist(5)));
  } else {
    // aten::col2im.out(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_col2im_out = [](Tensor out, const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::col2im_out(out, self, output_size, kernel_size, dilation, padding, stride);
    };
    return wrap(dispatch_col2im_out(_r.tensor(6), _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.intlist(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// elu
static PyObject * THPVariable_elu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "elu(Tensor input, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(4)) {
    // aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
    auto dispatch_elu = [](const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::elu(self, alpha, scale, input_scale);
    };
    return wrap(dispatch_elu(_r.tensor(0), _r.scalar(1), _r.scalar(2), _r.scalar(3)));
  } else {
    // aten::elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_elu_out = [](Tensor out, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::elu_out(out, self, alpha, scale, input_scale);
    };
    return wrap(dispatch_elu_out(_r.tensor(4), _r.tensor(0), _r.scalar(1), _r.scalar(2), _r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// elu_
static PyObject * THPVariable_elu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "elu_(Tensor input, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)
  auto dispatch_elu_ = [](Tensor self, Scalar alpha, Scalar scale, Scalar input_scale) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::elu_(self, alpha, scale, input_scale);
  };
  return wrap(dispatch_elu_(_r.tensor(0), _r.scalar(1), _r.scalar(2), _r.scalar(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fractional_max_pool2d
static PyObject * THPVariable_fractional_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fractional_max_pool2d(Tensor input, IntArrayRef[2] kernel_size, IntArrayRef[2] output_size, Tensor random_samples, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(4)) {
    // aten::fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)
    auto dispatch_fractional_max_pool2d = [](const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::fractional_max_pool2d(self, kernel_size, output_size, random_samples);
    };
    return wrap(dispatch_fractional_max_pool2d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.tensor(3)));
  } else {
    // aten::fractional_max_pool2d.output(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(4);
    auto dispatch_fractional_max_pool2d_out = [](Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::fractional_max_pool2d_out(output, indices, self, kernel_size, output_size, random_samples);
    };
    return wrap(dispatch_fractional_max_pool2d_out(out[0], out[1], _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.tensor(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fractional_max_pool3d
static PyObject * THPVariable_fractional_max_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fractional_max_pool3d(Tensor input, IntArrayRef[3] kernel_size, IntArrayRef[3] output_size, Tensor random_samples, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(4)) {
    // aten::fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)
    auto dispatch_fractional_max_pool3d = [](const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::fractional_max_pool3d(self, kernel_size, output_size, random_samples);
    };
    return wrap(dispatch_fractional_max_pool3d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.tensor(3)));
  } else {
    // aten::fractional_max_pool3d.output(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(4);
    auto dispatch_fractional_max_pool3d_out = [](Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::fractional_max_pool3d_out(output, indices, self, kernel_size, output_size, random_samples);
    };
    return wrap(dispatch_fractional_max_pool3d_out(out[0], out[1], _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.tensor(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// gelu
static PyObject * THPVariable_gelu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gelu(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::gelu(Tensor self) -> Tensor
  auto dispatch_gelu = [](const Tensor & self) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::gelu(self);
  };
  return wrap(dispatch_gelu(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// glu
static PyObject * THPVariable_glu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "glu(Tensor input, int64_t dim=-1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(2)) {
    // aten::glu(Tensor self, int dim=-1) -> Tensor
    auto dispatch_glu = [](const Tensor & self, int64_t dim) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::glu(self, dim);
    };
    return wrap(dispatch_glu(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::glu.out(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_glu_out = [](Tensor out, const Tensor & self, int64_t dim) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::glu_out(out, self, dim);
    };
    return wrap(dispatch_glu_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hardtanh
static PyObject * THPVariable_hardtanh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardtanh(Tensor input, Scalar min_val=-1, Scalar max_val=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(3)) {
    // aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
    auto dispatch_hardtanh = [](const Tensor & self, Scalar min_val, Scalar max_val) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hardtanh(self, min_val, max_val);
    };
    return wrap(dispatch_hardtanh(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
  } else {
    // aten::hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_hardtanh_out = [](Tensor out, const Tensor & self, Scalar min_val, Scalar max_val) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hardtanh_out(out, self, min_val, max_val);
    };
    return wrap(dispatch_hardtanh_out(_r.tensor(3), _r.tensor(0), _r.scalar(1), _r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hardtanh_
static PyObject * THPVariable_hardtanh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardtanh_(Tensor input, Scalar min_val=-1, Scalar max_val=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)
  auto dispatch_hardtanh_ = [](Tensor self, Scalar min_val, Scalar max_val) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::hardtanh_(self, min_val, max_val);
  };
  return wrap(dispatch_hardtanh_(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// im2col
static PyObject * THPVariable_im2col(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "im2col(Tensor input, IntArrayRef[2] kernel_size, IntArrayRef[2] dilation, IntArrayRef[2] padding, IntArrayRef[2] stride, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(5)) {
    // aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
    auto dispatch_im2col = [](const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::im2col(self, kernel_size, dilation, padding, stride);
    };
    return wrap(dispatch_im2col(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4)));
  } else {
    // aten::im2col.out(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_im2col_out = [](Tensor out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::im2col_out(out, self, kernel_size, dilation, padding, stride);
    };
    return wrap(dispatch_im2col_out(_r.tensor(5), _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// l1_loss
static PyObject * THPVariable_l1_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "l1_loss(Tensor input, Tensor target, int64_t reduction=at::Reduction::Mean, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(3)) {
    // aten::l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
    auto dispatch_l1_loss = [](const Tensor & self, const Tensor & target, int64_t reduction) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::l1_loss(self, target, reduction);
    };
    return wrap(dispatch_l1_loss(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  } else {
    // aten::l1_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_l1_loss_out = [](Tensor out, const Tensor & self, const Tensor & target, int64_t reduction) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::l1_loss_out(out, self, target, reduction);
    };
    return wrap(dispatch_l1_loss_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// leaky_relu
static PyObject * THPVariable_leaky_relu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "leaky_relu(Tensor input, Scalar negative_slope=0.01, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(2)) {
    // aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
    auto dispatch_leaky_relu = [](const Tensor & self, Scalar negative_slope) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::leaky_relu(self, negative_slope);
    };
    return wrap(dispatch_leaky_relu(_r.tensor(0), _r.scalar(1)));
  } else {
    // aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_leaky_relu_out = [](Tensor out, const Tensor & self, Scalar negative_slope) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::leaky_relu_out(out, self, negative_slope);
    };
    return wrap(dispatch_leaky_relu_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// leaky_relu_
static PyObject * THPVariable_leaky_relu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "leaky_relu_(Tensor input, Scalar negative_slope=0.01)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)
  auto dispatch_leaky_relu_ = [](Tensor self, Scalar negative_slope) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::leaky_relu_(self, negative_slope);
  };
  return wrap(dispatch_leaky_relu_(_r.tensor(0), _r.scalar(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linear
static PyObject * THPVariable_linear(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linear(Tensor input, Tensor weight, Tensor? bias=None)",
  }, /*traceable=*/false);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
  auto dispatch_linear = [](const Tensor & input, const Tensor & weight, const Tensor & bias) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::linear(input, weight, bias);
  };
  return wrap(dispatch_linear(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log_sigmoid
static PyObject * THPVariable_log_sigmoid(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log_sigmoid(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(1)) {
    // aten::log_sigmoid(Tensor self) -> Tensor
    auto dispatch_log_sigmoid = [](const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::log_sigmoid(self);
    };
    return wrap(dispatch_log_sigmoid(_r.tensor(0)));
  } else {
    // aten::log_sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_log_sigmoid_out = [](Tensor out, const Tensor & self) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::log_sigmoid_out(out, self);
    };
    return wrap(dispatch_log_sigmoid_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_pool2d_with_indices
static PyObject * THPVariable_max_pool2d_with_indices(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool2d_with_indices(Tensor input, IntArrayRef[2] kernel_size, IntArrayRef[2] stride=None, IntArrayRef[2] padding=0, IntArrayRef[2] dilation=1, bool ceil_mode=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(6)) {
    // aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
    auto dispatch_max_pool2d_with_indices = [](const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
    };
    return wrap(dispatch_max_pool2d_with_indices(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  } else {
    // aten::max_pool2d_with_indices.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(6);
    auto dispatch_max_pool2d_with_indices_out = [](Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::max_pool2d_with_indices_out(out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
    };
    return wrap(dispatch_max_pool2d_with_indices_out(out[0], out[1], _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_pool3d_with_indices
static PyObject * THPVariable_max_pool3d_with_indices(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool3d_with_indices(Tensor input, IntArrayRef[3] kernel_size, IntArrayRef[3] stride=None, IntArrayRef[3] padding=0, IntArrayRef[3] dilation=1, bool ceil_mode=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(6)) {
    // aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
    auto dispatch_max_pool3d_with_indices = [](const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
    };
    return wrap(dispatch_max_pool3d_with_indices(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  } else {
    // aten::max_pool3d_with_indices.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(6);
    auto dispatch_max_pool3d_with_indices_out = [](Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> std::tuple<Tensor,Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::max_pool3d_with_indices_out(out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
    };
    return wrap(dispatch_max_pool3d_with_indices_out(out[0], out[1], _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_unpool2d
static PyObject * THPVariable_max_unpool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_unpool2d(Tensor input, Tensor indices, IntArrayRef[2] output_size, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(3)) {
    // aten::max_unpool2d(Tensor self, Tensor indices, int[2] output_size) -> Tensor
    auto dispatch_max_unpool2d = [](const Tensor & self, const Tensor & indices, IntArrayRef output_size) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::max_unpool2d(self, indices, output_size);
    };
    return wrap(dispatch_max_unpool2d(_r.tensor(0), _r.tensor(1), _r.intlist(2)));
  } else {
    // aten::max_unpool2d.out(Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_max_unpool2d_out = [](Tensor out, const Tensor & self, const Tensor & indices, IntArrayRef output_size) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::max_unpool2d_out(out, self, indices, output_size);
    };
    return wrap(dispatch_max_unpool2d_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.intlist(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_unpool3d
static PyObject * THPVariable_max_unpool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_unpool3d(Tensor input, Tensor indices, IntArrayRef[3] output_size, IntArrayRef[3] stride, IntArrayRef[3] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(5)) {
    // aten::max_unpool3d(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor
    auto dispatch_max_unpool3d = [](const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::max_unpool3d(self, indices, output_size, stride, padding);
    };
    return wrap(dispatch_max_unpool3d(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3), _r.intlist(4)));
  } else {
    // aten::max_unpool3d.out(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_max_unpool3d_out = [](Tensor out, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::max_unpool3d_out(out, self, indices, output_size, stride, padding);
    };
    return wrap(dispatch_max_unpool3d_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3), _r.intlist(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_linear
static PyObject * THPVariable_mkldnn_linear(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_linear(Tensor input, Tensor weight, Tensor? bias=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::mkldnn_linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
  auto dispatch_mkldnn_linear = [](const Tensor & input, const Tensor & weight, const Tensor & bias) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::mkldnn_linear(input, weight, bias);
  };
  return wrap(dispatch_mkldnn_linear(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_reorder_conv2d_weight
static PyObject * THPVariable_mkldnn_reorder_conv2d_weight(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_reorder_conv2d_weight(Tensor input, IntArrayRef[2] padding=0, IntArrayRef[2] stride=1, IntArrayRef[2] dilation=1, int64_t groups=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::mkldnn_reorder_conv2d_weight(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1) -> Tensor
  auto dispatch_mkldnn_reorder_conv2d_weight = [](const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::mkldnn_reorder_conv2d_weight(self, padding, stride, dilation, groups);
  };
  return wrap(dispatch_mkldnn_reorder_conv2d_weight(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mse_loss
static PyObject * THPVariable_mse_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mse_loss(Tensor input, Tensor target, int64_t reduction=at::Reduction::Mean, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(3)) {
    // aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
    auto dispatch_mse_loss = [](const Tensor & self, const Tensor & target, int64_t reduction) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mse_loss(self, target, reduction);
    };
    return wrap(dispatch_mse_loss(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  } else {
    // aten::mse_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_mse_loss_out = [](Tensor out, const Tensor & self, const Tensor & target, int64_t reduction) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mse_loss_out(out, self, target, reduction);
    };
    return wrap(dispatch_mse_loss_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// multi_margin_loss
static PyObject * THPVariable_multi_margin_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "multi_margin_loss(Tensor input, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int64_t reduction=at::Reduction::Mean, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(6)) {
    // aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor
    auto dispatch_multi_margin_loss = [](const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::multi_margin_loss(self, target, p, margin, weight, reduction);
    };
    return wrap(dispatch_multi_margin_loss(_r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3), _r.tensor(4), _r.toInt64(5)));
  } else {
    // aten::multi_margin_loss.out(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_multi_margin_loss_out = [](Tensor out, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::multi_margin_loss_out(out, self, target, p, margin, weight, reduction);
    };
    return wrap(dispatch_multi_margin_loss_out(_r.tensor(6), _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3), _r.tensor(4), _r.toInt64(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// multilabel_margin_loss
static PyObject * THPVariable_multilabel_margin_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "multilabel_margin_loss(Tensor input, Tensor target, int64_t reduction=at::Reduction::Mean, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(3)) {
    // aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
    auto dispatch_multilabel_margin_loss = [](const Tensor & self, const Tensor & target, int64_t reduction) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::multilabel_margin_loss(self, target, reduction);
    };
    return wrap(dispatch_multilabel_margin_loss(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  } else {
    // aten::multilabel_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_multilabel_margin_loss_out = [](Tensor out, const Tensor & self, const Tensor & target, int64_t reduction) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::multilabel_margin_loss_out(out, self, target, reduction);
    };
    return wrap(dispatch_multilabel_margin_loss_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nll_loss
static PyObject * THPVariable_nll_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nll_loss(Tensor input, Tensor target, Tensor? weight=None, int64_t reduction=at::Reduction::Mean, int64_t ignore_index=-100, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(5)) {
    // aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor
    auto dispatch_nll_loss = [](const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::nll_loss(self, target, weight, reduction, ignore_index);
    };
    return wrap(dispatch_nll_loss(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4)));
  } else {
    // aten::nll_loss.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_nll_loss_out = [](Tensor out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::nll_loss_out(out, self, target, weight, reduction, ignore_index);
    };
    return wrap(dispatch_nll_loss_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nll_loss2d
static PyObject * THPVariable_nll_loss2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nll_loss2d(Tensor input, Tensor target, Tensor? weight=None, int64_t reduction=at::Reduction::Mean, int64_t ignore_index=-100, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(5)) {
    // aten::nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor
    auto dispatch_nll_loss2d = [](const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::nll_loss2d(self, target, weight, reduction, ignore_index);
    };
    return wrap(dispatch_nll_loss2d(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4)));
  } else {
    // aten::nll_loss2d.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_nll_loss2d_out = [](Tensor out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::nll_loss2d_out(out, self, target, weight, reduction, ignore_index);
    };
    return wrap(dispatch_nll_loss2d_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// one_hot
static PyObject * THPVariable_one_hot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "one_hot(Tensor input, int64_t num_classes=-1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::one_hot(Tensor self, int num_classes=-1) -> Tensor
  auto dispatch_one_hot = [](const Tensor & self, int64_t num_classes) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::one_hot(self, num_classes);
  };
  return wrap(dispatch_one_hot(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reflection_pad1d
static PyObject * THPVariable_reflection_pad1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reflection_pad1d(Tensor input, IntArrayRef[2] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(2)) {
    // aten::reflection_pad1d(Tensor self, int[2] padding) -> Tensor
    auto dispatch_reflection_pad1d = [](const Tensor & self, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::reflection_pad1d(self, padding);
    };
    return wrap(dispatch_reflection_pad1d(_r.tensor(0), _r.intlist(1)));
  } else {
    // aten::reflection_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_reflection_pad1d_out = [](Tensor out, const Tensor & self, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::reflection_pad1d_out(out, self, padding);
    };
    return wrap(dispatch_reflection_pad1d_out(_r.tensor(2), _r.tensor(0), _r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reflection_pad2d
static PyObject * THPVariable_reflection_pad2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reflection_pad2d(Tensor input, IntArrayRef[4] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(2)) {
    // aten::reflection_pad2d(Tensor self, int[4] padding) -> Tensor
    auto dispatch_reflection_pad2d = [](const Tensor & self, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::reflection_pad2d(self, padding);
    };
    return wrap(dispatch_reflection_pad2d(_r.tensor(0), _r.intlist(1)));
  } else {
    // aten::reflection_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_reflection_pad2d_out = [](Tensor out, const Tensor & self, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::reflection_pad2d_out(out, self, padding);
    };
    return wrap(dispatch_reflection_pad2d_out(_r.tensor(2), _r.tensor(0), _r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// replication_pad1d
static PyObject * THPVariable_replication_pad1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "replication_pad1d(Tensor input, IntArrayRef[2] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(2)) {
    // aten::replication_pad1d(Tensor self, int[2] padding) -> Tensor
    auto dispatch_replication_pad1d = [](const Tensor & self, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::replication_pad1d(self, padding);
    };
    return wrap(dispatch_replication_pad1d(_r.tensor(0), _r.intlist(1)));
  } else {
    // aten::replication_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_replication_pad1d_out = [](Tensor out, const Tensor & self, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::replication_pad1d_out(out, self, padding);
    };
    return wrap(dispatch_replication_pad1d_out(_r.tensor(2), _r.tensor(0), _r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// replication_pad2d
static PyObject * THPVariable_replication_pad2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "replication_pad2d(Tensor input, IntArrayRef[4] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(2)) {
    // aten::replication_pad2d(Tensor self, int[4] padding) -> Tensor
    auto dispatch_replication_pad2d = [](const Tensor & self, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::replication_pad2d(self, padding);
    };
    return wrap(dispatch_replication_pad2d(_r.tensor(0), _r.intlist(1)));
  } else {
    // aten::replication_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_replication_pad2d_out = [](Tensor out, const Tensor & self, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::replication_pad2d_out(out, self, padding);
    };
    return wrap(dispatch_replication_pad2d_out(_r.tensor(2), _r.tensor(0), _r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// replication_pad3d
static PyObject * THPVariable_replication_pad3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "replication_pad3d(Tensor input, IntArrayRef[6] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(2)) {
    // aten::replication_pad3d(Tensor self, int[6] padding) -> Tensor
    auto dispatch_replication_pad3d = [](const Tensor & self, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::replication_pad3d(self, padding);
    };
    return wrap(dispatch_replication_pad3d(_r.tensor(0), _r.intlist(1)));
  } else {
    // aten::replication_pad3d.out(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_replication_pad3d_out = [](Tensor out, const Tensor & self, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::replication_pad3d_out(out, self, padding);
    };
    return wrap(dispatch_replication_pad3d_out(_r.tensor(2), _r.tensor(0), _r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rrelu_with_noise
static PyObject * THPVariable_rrelu_with_noise(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu_with_noise(Tensor input, Tensor noise, Scalar lower=0.125, Scalar upper=0.333333333333, bool training=False, Generator generator=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(6)) {
    // aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
    auto dispatch_rrelu_with_noise = [](const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::rrelu_with_noise(self, noise, lower, upper, training, generator);
    };
    return wrap(dispatch_rrelu_with_noise(_r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3), _r.toBool(4), _r.generator(5)));
  } else {
    // aten::rrelu_with_noise.out(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_rrelu_with_noise_out = [](Tensor out, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::rrelu_with_noise_out(out, self, noise, lower, upper, training, generator);
    };
    return wrap(dispatch_rrelu_with_noise_out(_r.tensor(6), _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3), _r.toBool(4), _r.generator(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rrelu_with_noise_
static PyObject * THPVariable_rrelu_with_noise_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu_with_noise_(Tensor input, Tensor noise, Scalar lower=0.125, Scalar upper=0.333333333333, bool training=False, Generator generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::rrelu_with_noise_(Tensor(a!) self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)
  auto dispatch_rrelu_with_noise_ = [](Tensor self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::rrelu_with_noise_(self, noise, lower, upper, training, generator);
  };
  return wrap(dispatch_rrelu_with_noise_(_r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3), _r.toBool(4), _r.generator(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slow_conv3d
static PyObject * THPVariable_slow_conv3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slow_conv3d(Tensor input, Tensor weight, IntArrayRef[3] kernel_size, Tensor? bias=None, IntArrayRef[3] stride=1, IntArrayRef[3] padding=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(6)) {
    // aten::slow_conv3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0) -> Tensor
    auto dispatch_slow_conv3d = [](const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slow_conv3d(self, weight, kernel_size, bias, stride, padding);
    };
    return wrap(dispatch_slow_conv3d(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.tensor(3), _r.intlist(4), _r.intlist(5)));
  } else {
    // aten::slow_conv3d.out(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_slow_conv3d_out = [](Tensor out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slow_conv3d_out(out, self, weight, kernel_size, bias, stride, padding);
    };
    return wrap(dispatch_slow_conv3d_out(_r.tensor(6), _r.tensor(0), _r.tensor(1), _r.intlist(2), _r.tensor(3), _r.intlist(4), _r.intlist(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slow_conv_dilated2d
static PyObject * THPVariable_slow_conv_dilated2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slow_conv_dilated2d(Tensor input, Tensor weight, IntArrayRef[2] kernel_size, Tensor? bias=None, IntArrayRef[2] stride=1, IntArrayRef[2] padding=0, IntArrayRef[2] dilation=1)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::slow_conv_dilated2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor
  auto dispatch_slow_conv_dilated2d = [](const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
  };
  return wrap(dispatch_slow_conv_dilated2d(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.tensor(3), _r.intlist(4), _r.intlist(5), _r.intlist(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slow_conv_dilated3d
static PyObject * THPVariable_slow_conv_dilated3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slow_conv_dilated3d(Tensor input, Tensor weight, IntArrayRef[3] kernel_size, Tensor? bias=None, IntArrayRef[3] stride=1, IntArrayRef[3] padding=0, IntArrayRef[3] dilation=1)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  // aten::slow_conv_dilated3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1) -> Tensor
  auto dispatch_slow_conv_dilated3d = [](const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::slow_conv_dilated3d(self, weight, kernel_size, bias, stride, padding, dilation);
  };
  return wrap(dispatch_slow_conv_dilated3d(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.tensor(3), _r.intlist(4), _r.intlist(5), _r.intlist(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slow_conv_transpose2d
static PyObject * THPVariable_slow_conv_transpose2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slow_conv_transpose2d(Tensor input, Tensor weight, IntArrayRef[2] kernel_size, Tensor? bias=None, IntArrayRef[2] stride=1, IntArrayRef[2] padding=0, IntArrayRef[2] output_padding=0, IntArrayRef[2] dilation=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(8)) {
    // aten::slow_conv_transpose2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1) -> Tensor
    auto dispatch_slow_conv_transpose2d = [](const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
    };
    return wrap(dispatch_slow_conv_transpose2d(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.tensor(3), _r.intlist(4), _r.intlist(5), _r.intlist(6), _r.intlist(7)));
  } else {
    // aten::slow_conv_transpose2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_slow_conv_transpose2d_out = [](Tensor out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slow_conv_transpose2d_out(out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
    };
    return wrap(dispatch_slow_conv_transpose2d_out(_r.tensor(8), _r.tensor(0), _r.tensor(1), _r.intlist(2), _r.tensor(3), _r.intlist(4), _r.intlist(5), _r.intlist(6), _r.intlist(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slow_conv_transpose3d
static PyObject * THPVariable_slow_conv_transpose3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slow_conv_transpose3d(Tensor input, Tensor weight, IntArrayRef[3] kernel_size, Tensor? bias=None, IntArrayRef[3] stride=1, IntArrayRef[3] padding=0, IntArrayRef[3] output_padding=0, IntArrayRef[3] dilation=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(8)) {
    // aten::slow_conv_transpose3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1) -> Tensor
    auto dispatch_slow_conv_transpose3d = [](const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slow_conv_transpose3d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
    };
    return wrap(dispatch_slow_conv_transpose3d(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.tensor(3), _r.intlist(4), _r.intlist(5), _r.intlist(6), _r.intlist(7)));
  } else {
    // aten::slow_conv_transpose3d.out(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_slow_conv_transpose3d_out = [](Tensor out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slow_conv_transpose3d_out(out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
    };
    return wrap(dispatch_slow_conv_transpose3d_out(_r.tensor(8), _r.tensor(0), _r.tensor(1), _r.intlist(2), _r.tensor(3), _r.intlist(4), _r.intlist(5), _r.intlist(6), _r.intlist(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// smooth_l1_loss
static PyObject * THPVariable_smooth_l1_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "smooth_l1_loss(Tensor input, Tensor target, int64_t reduction=at::Reduction::Mean, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(3)) {
    // aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
    auto dispatch_smooth_l1_loss = [](const Tensor & self, const Tensor & target, int64_t reduction) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::smooth_l1_loss(self, target, reduction);
    };
    return wrap(dispatch_smooth_l1_loss(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  } else {
    // aten::smooth_l1_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_smooth_l1_loss_out = [](Tensor out, const Tensor & self, const Tensor & target, int64_t reduction) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::smooth_l1_loss_out(out, self, target, reduction);
    };
    return wrap(dispatch_smooth_l1_loss_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// soft_margin_loss
static PyObject * THPVariable_soft_margin_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "soft_margin_loss(Tensor input, Tensor target, int64_t reduction=at::Reduction::Mean, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(3)) {
    // aten::soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
    auto dispatch_soft_margin_loss = [](const Tensor & self, const Tensor & target, int64_t reduction) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::soft_margin_loss(self, target, reduction);
    };
    return wrap(dispatch_soft_margin_loss(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  } else {
    // aten::soft_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_soft_margin_loss_out = [](Tensor out, const Tensor & self, const Tensor & target, int64_t reduction) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::soft_margin_loss_out(out, self, target, reduction);
    };
    return wrap(dispatch_soft_margin_loss_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// softplus
static PyObject * THPVariable_softplus(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "softplus(Tensor input, Scalar beta=1, Scalar threshold=20, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(3)) {
    // aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor
    auto dispatch_softplus = [](const Tensor & self, Scalar beta, Scalar threshold) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::softplus(self, beta, threshold);
    };
    return wrap(dispatch_softplus(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
  } else {
    // aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_softplus_out = [](Tensor out, const Tensor & self, Scalar beta, Scalar threshold) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::softplus_out(out, self, beta, threshold);
    };
    return wrap(dispatch_softplus_out(_r.tensor(3), _r.tensor(0), _r.scalar(1), _r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// softshrink
static PyObject * THPVariable_softshrink(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "softshrink(Tensor input, Scalar lambd=0.5, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(2)) {
    // aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor
    auto dispatch_softshrink = [](const Tensor & self, Scalar lambd) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::softshrink(self, lambd);
    };
    return wrap(dispatch_softshrink(_r.tensor(0), _r.scalar(1)));
  } else {
    // aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_softshrink_out = [](Tensor out, const Tensor & self, Scalar lambd) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::softshrink_out(out, self, lambd);
    };
    return wrap(dispatch_softshrink_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// thnn_conv2d
static PyObject * THPVariable_thnn_conv2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "thnn_conv2d(Tensor input, Tensor weight, IntArrayRef[2] kernel_size, Tensor? bias=None, IntArrayRef[2] stride=1, IntArrayRef[2] padding=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(6)) {
    // aten::thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0) -> Tensor
    auto dispatch_thnn_conv2d = [](const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::thnn_conv2d(self, weight, kernel_size, bias, stride, padding);
    };
    return wrap(dispatch_thnn_conv2d(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.tensor(3), _r.intlist(4), _r.intlist(5)));
  } else {
    // aten::thnn_conv2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_thnn_conv2d_out = [](Tensor out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::thnn_conv2d_out(out, self, weight, kernel_size, bias, stride, padding);
    };
    return wrap(dispatch_thnn_conv2d_out(_r.tensor(6), _r.tensor(0), _r.tensor(1), _r.intlist(2), _r.tensor(3), _r.intlist(4), _r.intlist(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// thnn_conv_depthwise2d
static PyObject * THPVariable_thnn_conv_depthwise2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "thnn_conv_depthwise2d(Tensor input, Tensor weight, IntArrayRef[2] kernel_size, Tensor? bias=None, IntArrayRef[2] stride=1, IntArrayRef[2] padding=0, IntArrayRef[2] dilation=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(7)) {
    // aten::thnn_conv_depthwise2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor
    auto dispatch_thnn_conv_depthwise2d = [](const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::thnn_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation);
    };
    return wrap(dispatch_thnn_conv_depthwise2d(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.tensor(3), _r.intlist(4), _r.intlist(5), _r.intlist(6)));
  } else {
    // aten::thnn_conv_depthwise2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_thnn_conv_depthwise2d_out = [](Tensor out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::thnn_conv_depthwise2d_out(out, self, weight, kernel_size, bias, stride, padding, dilation);
    };
    return wrap(dispatch_thnn_conv_depthwise2d_out(_r.tensor(7), _r.tensor(0), _r.tensor(1), _r.intlist(2), _r.tensor(3), _r.intlist(4), _r.intlist(5), _r.intlist(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// upsample_bicubic2d
static PyObject * THPVariable_upsample_bicubic2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_bicubic2d(Tensor input, IntArrayRef[2] output_size, bool align_corners, double? scales_h=None, double? scales_w=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(5)) {
    // aten::upsample_bicubic2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
    auto dispatch_upsample_bicubic2d = [](const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::upsample_bicubic2d(self, output_size, align_corners, scales_h, scales_w);
    };
    return wrap(dispatch_upsample_bicubic2d(_r.tensor(0), _r.intlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
  } else {
    // aten::upsample_bicubic2d.out(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_upsample_bicubic2d_out = [](Tensor out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::upsample_bicubic2d_out(out, self, output_size, align_corners, scales_h, scales_w);
    };
    return wrap(dispatch_upsample_bicubic2d_out(_r.tensor(5), _r.tensor(0), _r.intlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// upsample_bilinear2d
static PyObject * THPVariable_upsample_bilinear2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_bilinear2d(Tensor input, IntArrayRef[2] output_size, bool align_corners, double? scales_h=None, double? scales_w=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(5)) {
    // aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
    auto dispatch_upsample_bilinear2d = [](const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::upsample_bilinear2d(self, output_size, align_corners, scales_h, scales_w);
    };
    return wrap(dispatch_upsample_bilinear2d(_r.tensor(0), _r.intlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
  } else {
    // aten::upsample_bilinear2d.out(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_upsample_bilinear2d_out = [](Tensor out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::upsample_bilinear2d_out(out, self, output_size, align_corners, scales_h, scales_w);
    };
    return wrap(dispatch_upsample_bilinear2d_out(_r.tensor(5), _r.tensor(0), _r.intlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// upsample_linear1d
static PyObject * THPVariable_upsample_linear1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_linear1d(Tensor input, IntArrayRef[1] output_size, bool align_corners, double? scales=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(4)) {
    // aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners, float? scales=None) -> Tensor
    auto dispatch_upsample_linear1d = [](const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::upsample_linear1d(self, output_size, align_corners, scales);
    };
    return wrap(dispatch_upsample_linear1d(_r.tensor(0), _r.intlist(1), _r.toBool(2), _r.toDoubleOptional(3)));
  } else {
    // aten::upsample_linear1d.out(Tensor self, int[1] output_size, bool align_corners, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_upsample_linear1d_out = [](Tensor out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::upsample_linear1d_out(out, self, output_size, align_corners, scales);
    };
    return wrap(dispatch_upsample_linear1d_out(_r.tensor(4), _r.tensor(0), _r.intlist(1), _r.toBool(2), _r.toDoubleOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// upsample_nearest1d
static PyObject * THPVariable_upsample_nearest1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_nearest1d(Tensor input, IntArrayRef[1] output_size, double? scales=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(3)) {
    // aten::upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> Tensor
    auto dispatch_upsample_nearest1d = [](const Tensor & self, IntArrayRef output_size, c10::optional<double> scales) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::upsample_nearest1d(self, output_size, scales);
    };
    return wrap(dispatch_upsample_nearest1d(_r.tensor(0), _r.intlist(1), _r.toDoubleOptional(2)));
  } else {
    // aten::upsample_nearest1d.out(Tensor self, int[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_upsample_nearest1d_out = [](Tensor out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::upsample_nearest1d_out(out, self, output_size, scales);
    };
    return wrap(dispatch_upsample_nearest1d_out(_r.tensor(3), _r.tensor(0), _r.intlist(1), _r.toDoubleOptional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// upsample_nearest2d
static PyObject * THPVariable_upsample_nearest2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_nearest2d(Tensor input, IntArrayRef[2] output_size, double? scales_h=None, double? scales_w=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(4)) {
    // aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor
    auto dispatch_upsample_nearest2d = [](const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::upsample_nearest2d(self, output_size, scales_h, scales_w);
    };
    return wrap(dispatch_upsample_nearest2d(_r.tensor(0), _r.intlist(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3)));
  } else {
    // aten::upsample_nearest2d.out(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_upsample_nearest2d_out = [](Tensor out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::upsample_nearest2d_out(out, self, output_size, scales_h, scales_w);
    };
    return wrap(dispatch_upsample_nearest2d_out(_r.tensor(4), _r.tensor(0), _r.intlist(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// upsample_nearest3d
static PyObject * THPVariable_upsample_nearest3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_nearest3d(Tensor input, IntArrayRef[3] output_size, double? scales_d=None, double? scales_h=None, double? scales_w=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(5)) {
    // aten::upsample_nearest3d(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
    auto dispatch_upsample_nearest3d = [](const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::upsample_nearest3d(self, output_size, scales_d, scales_h, scales_w);
    };
    return wrap(dispatch_upsample_nearest3d(_r.tensor(0), _r.intlist(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
  } else {
    // aten::upsample_nearest3d.out(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_upsample_nearest3d_out = [](Tensor out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::upsample_nearest3d_out(out, self, output_size, scales_d, scales_h, scales_w);
    };
    return wrap(dispatch_upsample_nearest3d_out(_r.tensor(5), _r.tensor(0), _r.intlist(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// upsample_trilinear3d
static PyObject * THPVariable_upsample_trilinear3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_trilinear3d(Tensor input, IntArrayRef[3] output_size, bool align_corners, double? scales_d=None, double? scales_h=None, double? scales_w=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  if (_r.isNone(6)) {
    // aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
    auto dispatch_upsample_trilinear3d = [](const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::upsample_trilinear3d(self, output_size, align_corners, scales_d, scales_h, scales_w);
    };
    return wrap(dispatch_upsample_trilinear3d(_r.tensor(0), _r.intlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4), _r.toDoubleOptional(5)));
  } else {
    // aten::upsample_trilinear3d.out(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
    auto dispatch_upsample_trilinear3d_out = [](Tensor out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) -> Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::upsample_trilinear3d_out(out, self, output_size, align_corners, scales_d, scales_h, scales_w);
    };
    return wrap(dispatch_upsample_trilinear3d_out(_r.tensor(6), _r.tensor(0), _r.intlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4), _r.toDoubleOptional(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyMethodDef nn_functions[] = {
  {"_parse_to", (PyCFunction)(void(*)(void))THPVariable__parse_to, METH_VARARGS | METH_KEYWORDS, nullptr},
  {"adaptive_avg_pool2d", (PyCFunction)(void(*)(void))THPVariable_adaptive_avg_pool2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"adaptive_avg_pool3d", (PyCFunction)(void(*)(void))THPVariable_adaptive_avg_pool3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"adaptive_max_pool2d", (PyCFunction)(void(*)(void))THPVariable_adaptive_max_pool2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"adaptive_max_pool3d", (PyCFunction)(void(*)(void))THPVariable_adaptive_max_pool3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"avg_pool2d", (PyCFunction)(void(*)(void))THPVariable_avg_pool2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"avg_pool3d", (PyCFunction)(void(*)(void))THPVariable_avg_pool3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"binary_cross_entropy", (PyCFunction)(void(*)(void))THPVariable_binary_cross_entropy, METH_VARARGS | METH_KEYWORDS, NULL},
  {"col2im", (PyCFunction)(void(*)(void))THPVariable_col2im, METH_VARARGS | METH_KEYWORDS, NULL},
  {"elu", (PyCFunction)(void(*)(void))THPVariable_elu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"elu_", (PyCFunction)(void(*)(void))THPVariable_elu_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"fractional_max_pool2d", (PyCFunction)(void(*)(void))THPVariable_fractional_max_pool2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"fractional_max_pool3d", (PyCFunction)(void(*)(void))THPVariable_fractional_max_pool3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"gelu", (PyCFunction)(void(*)(void))THPVariable_gelu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"glu", (PyCFunction)(void(*)(void))THPVariable_glu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"hardtanh", (PyCFunction)(void(*)(void))THPVariable_hardtanh, METH_VARARGS | METH_KEYWORDS, NULL},
  {"hardtanh_", (PyCFunction)(void(*)(void))THPVariable_hardtanh_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"im2col", (PyCFunction)(void(*)(void))THPVariable_im2col, METH_VARARGS | METH_KEYWORDS, NULL},
  {"l1_loss", (PyCFunction)(void(*)(void))THPVariable_l1_loss, METH_VARARGS | METH_KEYWORDS, NULL},
  {"leaky_relu", (PyCFunction)(void(*)(void))THPVariable_leaky_relu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"leaky_relu_", (PyCFunction)(void(*)(void))THPVariable_leaky_relu_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"linear", (PyCFunction)(void(*)(void))THPVariable_linear, METH_VARARGS | METH_KEYWORDS, NULL},
  {"log_sigmoid", (PyCFunction)(void(*)(void))THPVariable_log_sigmoid, METH_VARARGS | METH_KEYWORDS, NULL},
  {"max_pool2d_with_indices", (PyCFunction)(void(*)(void))THPVariable_max_pool2d_with_indices, METH_VARARGS | METH_KEYWORDS, NULL},
  {"max_pool3d_with_indices", (PyCFunction)(void(*)(void))THPVariable_max_pool3d_with_indices, METH_VARARGS | METH_KEYWORDS, NULL},
  {"max_unpool2d", (PyCFunction)(void(*)(void))THPVariable_max_unpool2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"max_unpool3d", (PyCFunction)(void(*)(void))THPVariable_max_unpool3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mkldnn_linear", (PyCFunction)(void(*)(void))THPVariable_mkldnn_linear, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mkldnn_reorder_conv2d_weight", (PyCFunction)(void(*)(void))THPVariable_mkldnn_reorder_conv2d_weight, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mse_loss", (PyCFunction)(void(*)(void))THPVariable_mse_loss, METH_VARARGS | METH_KEYWORDS, NULL},
  {"multi_margin_loss", (PyCFunction)(void(*)(void))THPVariable_multi_margin_loss, METH_VARARGS | METH_KEYWORDS, NULL},
  {"multilabel_margin_loss", (PyCFunction)(void(*)(void))THPVariable_multilabel_margin_loss, METH_VARARGS | METH_KEYWORDS, NULL},
  {"nll_loss", (PyCFunction)(void(*)(void))THPVariable_nll_loss, METH_VARARGS | METH_KEYWORDS, NULL},
  {"nll_loss2d", (PyCFunction)(void(*)(void))THPVariable_nll_loss2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"one_hot", (PyCFunction)(void(*)(void))THPVariable_one_hot, METH_VARARGS | METH_KEYWORDS, NULL},
  {"reflection_pad1d", (PyCFunction)(void(*)(void))THPVariable_reflection_pad1d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"reflection_pad2d", (PyCFunction)(void(*)(void))THPVariable_reflection_pad2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"replication_pad1d", (PyCFunction)(void(*)(void))THPVariable_replication_pad1d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"replication_pad2d", (PyCFunction)(void(*)(void))THPVariable_replication_pad2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"replication_pad3d", (PyCFunction)(void(*)(void))THPVariable_replication_pad3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"rrelu_with_noise", (PyCFunction)(void(*)(void))THPVariable_rrelu_with_noise, METH_VARARGS | METH_KEYWORDS, NULL},
  {"rrelu_with_noise_", (PyCFunction)(void(*)(void))THPVariable_rrelu_with_noise_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"slow_conv3d", (PyCFunction)(void(*)(void))THPVariable_slow_conv3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"slow_conv_dilated2d", (PyCFunction)(void(*)(void))THPVariable_slow_conv_dilated2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"slow_conv_dilated3d", (PyCFunction)(void(*)(void))THPVariable_slow_conv_dilated3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"slow_conv_transpose2d", (PyCFunction)(void(*)(void))THPVariable_slow_conv_transpose2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"slow_conv_transpose3d", (PyCFunction)(void(*)(void))THPVariable_slow_conv_transpose3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"smooth_l1_loss", (PyCFunction)(void(*)(void))THPVariable_smooth_l1_loss, METH_VARARGS | METH_KEYWORDS, NULL},
  {"soft_margin_loss", (PyCFunction)(void(*)(void))THPVariable_soft_margin_loss, METH_VARARGS | METH_KEYWORDS, NULL},
  {"softplus", (PyCFunction)(void(*)(void))THPVariable_softplus, METH_VARARGS | METH_KEYWORDS, NULL},
  {"softshrink", (PyCFunction)(void(*)(void))THPVariable_softshrink, METH_VARARGS | METH_KEYWORDS, NULL},
  {"thnn_conv2d", (PyCFunction)(void(*)(void))THPVariable_thnn_conv2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"thnn_conv_depthwise2d", (PyCFunction)(void(*)(void))THPVariable_thnn_conv_depthwise2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"upsample_bicubic2d", (PyCFunction)(void(*)(void))THPVariable_upsample_bicubic2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"upsample_bilinear2d", (PyCFunction)(void(*)(void))THPVariable_upsample_bilinear2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"upsample_linear1d", (PyCFunction)(void(*)(void))THPVariable_upsample_linear1d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"upsample_nearest1d", (PyCFunction)(void(*)(void))THPVariable_upsample_nearest1d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"upsample_nearest2d", (PyCFunction)(void(*)(void))THPVariable_upsample_nearest2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"upsample_nearest3d", (PyCFunction)(void(*)(void))THPVariable_upsample_nearest3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"upsample_trilinear3d", (PyCFunction)(void(*)(void))THPVariable_upsample_trilinear3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {NULL}
};

void initNNFunctions(PyObject* module) {
#if PY_MAJOR_VERSION == 2
  PyObject* nn = Py_InitModule("torch._C._nn", nn_functions);
  Py_XINCREF(nn);  // Py_InitModule returns "borrowed" reference
#else
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._nn",
     NULL,
     -1,
     nn_functions
  };
  PyObject* nn = PyModule_Create(&def);
#endif
  if (!nn) {
    throw python_error();
  }
  // steals a reference to nn
  if (PyModule_AddObject(module, "_nn", nn) != 0) {
    throw python_error();
  }
}

}} // namespace torch::autograd
