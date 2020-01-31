#include <torch/csrc/autograd/python_variable_indexing.h>

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/THP_export.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/utils/tensor_types.h>

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/native/TensorIndexing.h>

#include <vector>
#include <tuple>

using namespace at;
using namespace at::indexing;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

Py_ssize_t THPVariable_length(PyObject* self) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.dim() == 0) {
    return 0;
  }
  return (Py_ssize_t)self_.size(0);
  END_HANDLE_TH_ERRORS_RET(-1)
}

// We allow indexing by integers, slices, ellipsis, None, Variables,
// and tuples of those types. We also handle bools as if they were a
// Variable[ByteTensor].

[[noreturn]]
static void invalid_index(PyObject* obj) {
  throw IndexError(
    "only integers, slices (`:`), ellipsis (`...`), None and long or byte "
    "Variables are valid indices (got %s)", Py_TYPE(obj)->tp_name);
}

static inline Variable sequenceToVariable(c10::DispatchKey dispatch_key, PyObject* seq) {
  return torch::utils::indexing_tensor_from_data(dispatch_key, kLong, c10::nullopt, seq);
}

static inline bool treatSequenceAsTuple(PyObject* index) {
  if (PyTuple_Check(index)) {
    return true;
  }
  if (!PySequence_Check(index)) {
    return false;
  }
  // This uses a heuristics from NumPy for determining whether to treat
  // non-tuple sequences as if they were a tuple. From the NumPy code comments:
  //
  // "At this point, we're left with a non-tuple, non-array, sequence:
  //  typically, a list. We use some somewhat-arbitrary heuristics from here
  //  onwards to decided whether to treat that list as a single index, or a
  //  list of indices. Backwards compatibility only takes effect for short
  //  sequences - otherwise we treat it like any other scalar."
  auto n = PySequence_Size(index);
  if (n < 0) {
    // Negative size indicates a Python error in the PySequence_Size call.
    PyErr_Clear();
    return false;
  }
  if (n >= 32) {
    return false;
  }
  for (Py_ssize_t i = 0; i < n; i++) {
    auto obj = THPObjectPtr{PySequence_GetItem(index, i)};
    if (!obj.get()) {
      PyErr_Clear();
      return false;
    }
    if (THPVariable_Check(obj.get()) || PySequence_Check(obj.get()) || PySlice_Check(obj.get())) {
      return true;
    }
    if (obj.get() == Py_Ellipsis || obj.get() == Py_None) {
      return true;
    }
  }
  return false;
}

static inline THPObjectPtr wrapTuple(PyObject* index) {
  THPObjectPtr res;
  if (treatSequenceAsTuple(index)) {
    res = PySequence_Tuple(index);
  } else {
    res = PyTuple_Pack(1, index); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
  }
  if (!res) throw python_error();
  return res;
}

static inline void indexToTensorIndexList(const Variable& self, PyObject* index, std::vector<TensorIndex>& tensor_index_list) {
  THPObjectPtr holder = wrapTuple(index);
  int64_t size = PyTuple_GET_SIZE(holder.get());

  tensor_index_list.reserve(size);

  for (int64_t i = 0; i < size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(holder.get(), i);
    if (THPUtils_checkLong(obj)) {
      if (THPVariable_Check(obj)) {
        tensor_index_list.emplace_back(TensorIndex(THPUtils_unpackLong(obj), THPVariable_Unpack(obj)));
      } else {
        tensor_index_list.emplace_back(TensorIndex(THPUtils_unpackLong(obj)));
      }
    } else if (PySlice_Check(obj)) {
      Py_ssize_t start, stop, step;
      if (!THPUtils_unpackSlice(obj, &start, &stop, &step)) {
        throw python_error();
      }

      PySliceObject* sliceobj = (PySliceObject*)obj;
      Tensor start_tensor, stop_tensor, step_tensor;
      if (THPVariable_Check(sliceobj->start)) {
        start_tensor = THPVariable_Unpack(sliceobj->start);
      }
      if (THPVariable_Check(sliceobj->stop)) {
        stop_tensor = THPVariable_Unpack(sliceobj->stop);
      }
      if (THPVariable_Check(sliceobj->step)) {
        step_tensor = THPVariable_Unpack(sliceobj->step);
      }

      tensor_index_list.emplace_back(TensorIndex({start, stop, step}, {start_tensor, stop_tensor, step_tensor}));
    } else if (obj == Py_Ellipsis) {
      tensor_index_list.emplace_back(TensorIndex(at::indexing::Ellipsis));
    } else if (obj == Py_None) {
      tensor_index_list.emplace_back(TensorIndex(at::indexing::None));
    } else if (PyBool_Check(obj)) {
      tensor_index_list.emplace_back(TensorIndex(obj == Py_True));
    } else if (THPVariable_Check(obj)) {
      tensor_index_list.emplace_back(TensorIndex(THPVariable_Unpack(obj)));
    } else if (PySequence_Check(obj)) {
      // TODO: Naughty naughty get out of jail free
      // (Fixing this means I have to fix the call chain though :/)
      tensor_index_list.emplace_back(TensorIndex(sequenceToVariable(legacyExtractDispatchKey(self), obj)));
    } else {
      auto ind = THPObjectPtr(PyNumber_Index(obj));
      if (!ind) {
        PyErr_Clear();
        invalid_index(obj);
      }
      if (THPVariable_Check(ind)) {
        tensor_index_list.emplace_back(TensorIndex(THPUtils_unpackLong(ind), THPVariable_Unpack(ind)));
      } else {
        tensor_index_list.emplace_back(TensorIndex(THPUtils_unpackLong(ind)));
      }
    }
  }
}

// static inline Tensor dispatch_index_no_gil(Tensor & self, const ArrayRef<TensorIndex>& tensor_index_list) {
//   pybind11::gil_scoped_release no_gil;
//   return at::indexing::get_item(self, tensor_index_list);
// }

static int64_t count_specified_dimensions(PyObject* index) {
  // Count the number of indexed dimensions (everything but ellipsis and None)
  int64_t count = 0;
  auto size = PyTuple_GET_SIZE(index); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
  for (Py_ssize_t i = 0; i < size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(index, i); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    if (THPVariable_Check(obj)) {
      auto& var = reinterpret_cast<THPVariable*>(obj)->cdata;
      if (var.scalar_type() == kByte || var.scalar_type() == kBool) {
        count += var.dim();
      } else {
        count++;
      }
    } else if (obj != Py_None && obj != Py_Ellipsis && obj != Py_True && obj != Py_False) { // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
      count++;
    }
  }
  return count;
}

static inline void unpackSliceAndExtractTensors(
  PyObject* obj,
  Py_ssize_t& start,
  Py_ssize_t& stop,
  Py_ssize_t& step,
  Tensor& start_tensor,
  Tensor& stop_tensor,
  Tensor& step_tensor) {
  if (!THPUtils_unpackSlice(obj, &start, &stop, &step)) {
    throw python_error();
  }

  PySliceObject* sliceobj = (PySliceObject*)obj;
  if (THPVariable_Check(sliceobj->start)) {
    start_tensor = THPVariable_Unpack(sliceobj->start);
  }
  if (THPVariable_Check(sliceobj->stop)) {
    stop_tensor = THPVariable_Unpack(sliceobj->stop);
  }
  if (THPVariable_Check(sliceobj->step)) {
    step_tensor = THPVariable_Unpack(sliceobj->step);
  }
}

static inline Variable applySlicing(const Variable& self, PyObject* index, variable_list& outIndices) {
  int64_t size = PyTuple_GET_SIZE(index); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
  int64_t dim = 0;
  int64_t specified_dims = count_specified_dimensions(index);

  if (specified_dims > self.dim()) {
    throw IndexError("too many indices for tensor of dimension %d", (int)self.dim());
  }

  Variable result = self;
  for (int64_t i = 0; i < size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(index, i); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    if (THPUtils_checkLong(obj)) {
      result = at::indexing::handleInteger(
        result,
        dim,
        THPUtils_unpackLong(obj),
        THPVariable_Check(obj) ? THPVariable_Unpack(obj) : Tensor(),
        i);
    } else if (PySlice_Check(obj)) {
      Py_ssize_t start, stop, step;
      if (!THPUtils_unpackSlice(obj, &start, &stop, &step)) {
        throw python_error();
      }

      PySliceObject* sliceobj = (PySliceObject*)obj;
      Tensor start_tensor, stop_tensor, step_tensor;
      if (THPVariable_Check(sliceobj->start)) {
        start_tensor = THPVariable_Unpack(sliceobj->start);
      }
      if (THPVariable_Check(sliceobj->stop)) {
        stop_tensor = THPVariable_Unpack(sliceobj->stop);
      }
      if (THPVariable_Check(sliceobj->step)) {
        step_tensor = THPVariable_Unpack(sliceobj->step);
      }
      result = at::indexing::handleSlice(
        result,
        dim,
        start,
        stop,
        step,
        start_tensor,
        stop_tensor,
        step_tensor);
    } else if (obj == Py_Ellipsis) {
      at::indexing::handleEllipsis(self, dim, specified_dims);
    } else if (obj == Py_None) {
      result = at::indexing::handleNone(result, dim);
    } else if (PyBool_Check(obj)) {
      result = at::indexing::handleBoolean(result, obj == Py_True, outIndices, dim);
    } else if (THPVariable_Check(obj)) {
      result = at::indexing::handleTensor(result, THPVariable_Unpack(obj), outIndices, dim, i);
    } else if (PySequence_Check(obj)) {
      // TODO: Naughty naughty get out of jail free
      // (Fixing this means I have to fix the call chain though :/)
      at::indexing::_record_tensor_index(sequenceToVariable(legacyExtractDispatchKey(self), obj), outIndices, dim);
    } else {
      auto idx = THPObjectPtr(PyNumber_Index(obj));
      if (!idx) {
        PyErr_Clear();
        invalid_index(obj);
      }
      result = at::indexing::handleInteger(
        result,
        dim,
        THPUtils_unpackLong(obj),
        THPVariable_Check(obj) ? THPVariable_Unpack(obj) : Tensor(),
        i);
    }
  }
  return result;
}

PyObject* THPVariable_getitem(PyObject* self, PyObject* index) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;

  // yf225 TODO: comment why we need this special case for 1-dim indexing
  OptionalDeviceGuard device_guard(device_of(self_));

  // handle simple types: integers, slices, ellipsis
  if (index == Py_None) {
    return wrap(at::indexing::handleNoneSingleDim(self_));
  } else if (index == Py_Ellipsis) {
    return wrap(at::indexing::handleEllipsisSingleDim(self_));
  } else if (THPUtils_checkLong(index)) {
    return wrap(at::indexing::handleIntegerSingleDim(
      self_,
      THPUtils_unpackLong(index),
      THPVariable_Check(index) ? THPVariable_Unpack(index) : Tensor()));
  } else if (PySlice_Check(index)) {
    Py_ssize_t start, stop, step;
    Tensor start_tensor, stop_tensor, step_tensor;
    unpackSliceAndExtractTensors(index, start, stop, step, start_tensor, stop_tensor, step_tensor);
    return wrap(at::indexing::handleSliceSingleDimGet(
      self_,
      start,
      stop,
      step,
      start_tensor,
      stop_tensor,
      step_tensor));
  }

  // wrap index in a tuple if it's not already one
  THPObjectPtr holder = wrapTuple(index);

  variable_list variableIndices;
  Variable sliced = applySlicing(self_, holder.get(), variableIndices);
  if (variableIndices.empty()) {
    if (sliced.is_same(self_)) {
      // ensure we return a shallow copy for things like x[...]
      sliced = at::alias(sliced);
    }
    return wrap(sliced);
  }

  // indexing by tensors ("advanced" indexing)
  return wrap(at::indexing::dispatch_index(sliced, variableIndices));

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static inline void dispatch_index_put_no_gil(Tensor & self, const ArrayRef<TensorIndex>& tensor_index_list, Tensor const & rhs) {
  pybind11::gil_scoped_release no_gil;
  at::indexing::set_item(self, tensor_index_list, rhs);
}

static inline void dispatch_index_put_no_gil(Tensor & self, const ArrayRef<TensorIndex>& tensor_index_list, Scalar v) {
  pybind11::gil_scoped_release no_gil;
  at::indexing::set_item(self, tensor_index_list, v);
}

int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* py_value) {
  HANDLE_TH_ERRORS
  if (py_value == nullptr) {
    throw TypeError("Tensor does not support deleting items");
  }
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;

  // yf225 TODO: decompose this as well!
  std::vector<TensorIndex> tensor_index_list;
  indexToTensorIndexList(self_, index, tensor_index_list);

  // yf225 TODO: do we need special case for 1-dim here as well?

  if (THPVariable_Check(py_value)) {
    dispatch_index_put_no_gil(self_, tensor_index_list, reinterpret_cast<THPVariable*>(py_value)->cdata);
    return 0;
  } else if (THPUtils_checkLong(py_value) || PyBool_Check(py_value)) {
    Scalar v = Scalar(THPUtils_unpackLong(py_value));
    dispatch_index_put_no_gil(self_, tensor_index_list, v);
    return 0;
  } else if (PyFloat_Check(py_value)) {
    Scalar v = Scalar(THPUtils_unpackDouble(py_value));
    dispatch_index_put_no_gil(self_, tensor_index_list, v);
    return 0;
  } else {
    throw TypeError(
      "can't assign a %s to a %s",
      Py_TYPE(py_value)->tp_name,
      torch::utils::options_to_string(self_.options()).c_str());
  }
  END_HANDLE_TH_ERRORS_RET(-1)
}

}} // namespace torch::autograd
