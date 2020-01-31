#pragma once

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/DeviceGuard.h>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/tracer.h>

namespace at {
namespace indexing {

const int64_t INDEX_MAX = std::numeric_limits<int64_t>::max();
const int64_t INDEX_MIN = std::numeric_limits<int64_t>::min();

enum class TensorIndexType { None, Ellipsis, Integer, Boolean, Slice, Tensor };

constexpr c10::nullopt_t None{c10::nullopt_t::init()};

struct CAFFE2_API EllipsisIndexType final { EllipsisIndexType() {} };
CAFFE2_API extern const EllipsisIndexType Ellipsis;

struct CAFFE2_API Slice final {
 public:
  Slice();
  Slice(
    int64_t start,
    int64_t stop,
    int64_t step,
    const Tensor& start_tensor,
    const Tensor& stop_tensor,
    const Tensor& step_tensor);

  inline int64_t start() const;
  inline int64_t stop() const;
  inline int64_t step() const;

  inline const Tensor& start_tensor() const;
  inline const Tensor& stop_tensor() const;
  inline const Tensor& step_tensor() const;

  inline bool has_start_tensor() const;
  inline bool has_stop_tensor() const;
  inline bool has_step_tensor() const;

 private:
  int64_t start_;
  int64_t stop_;
  int64_t step_;
  Tensor start_tensor_;
  Tensor stop_tensor_;
  Tensor step_tensor_;
};

Slice::Slice() {}

Slice::Slice(
    int64_t start,
    int64_t stop,
    int64_t step,
    const Tensor& start_tensor,
    const Tensor& stop_tensor,
    const Tensor& step_tensor)
  : start_(start),
    stop_(stop),
    step_(step),
    start_tensor_(start_tensor),
    stop_tensor_(stop_tensor),
    step_tensor_(step_tensor) {}

inline int64_t Slice::start() const {
  return start_;
}

inline int64_t Slice::stop() const {
  return stop_;
}

inline int64_t Slice::step() const {
  return step_;
}

inline const Tensor& Slice::start_tensor() const {
  return start_tensor_;
}

inline const Tensor& Slice::stop_tensor() const {
  return stop_tensor_;
}

inline const Tensor& Slice::step_tensor() const {
  return step_tensor_;
}

inline bool Slice::has_start_tensor() const {
  return start_tensor_.defined();
}

inline bool Slice::has_stop_tensor() const {
  return stop_tensor_.defined();
}

inline bool Slice::has_step_tensor() const {
  return step_tensor_.defined();
}

CAFFE2_API std::ostream& operator<<(std::ostream& stream, const Slice& slice);

// `at::indexing::TensorIndex` is used for converting C++ tensor indices such as
// `{None, "...", Ellipsis, 0, true, {1, None, 2}, torch::tensor({1, 2})}`
// into its equivalent `std::vector<TensorIndex>`, so that further tensor indexing
// operations can be performed using the supplied indices.
//
// There is one-to-one correspondence between Python and C++ tensor index types:
// Python                  | C++
// -----------------------------------------------------
// `None`                  | `at::indexing::None`
// `Ellipsis`              | `at::indexing::Ellipsis`
// `...`                   | `"..."`
// `123`                   | `123`
// `True` / `False`        | `true` / `false`
// `:`                     | `{}` / `{None, None}`
// `::`                    | `{}` / `{None, None, None}`
// `1:`                    | `{1, None}`
// `1::`                   | `{1, None, None}`
// `:3`                    | `{None, 3}`
// `:3:`                   | `{None, 3, None}`
// `::2`                   | `{None, None, 2}`
// `1:3`                   | `{1, 3}`
// `1::2`                  | `{1, None, 2}`
// `:3:2`                  | `{None, 3, 2}`
// `1:3:2`                 | `{1, 3, 2}`
// `torch.tensor([1, 2])`) | `torch::tensor({1, 2})`
struct CAFFE2_API TensorIndex final {
  // Case 1: `at::indexing::None`
  TensorIndex(c10::nullopt_t);

  // Case 2: "..." / `at::indexing::Ellipsis`
  TensorIndex(at::indexing::EllipsisIndexType);
  TensorIndex(const char *str);

  // Case 3: Integer value (the tensor form can optionally be provided)
  TensorIndex(int64_t integer, Tensor tensor = {});
  TensorIndex(int integer);

  // Case 4: Boolean value
  template <class T,
            class = typename std::enable_if<std::is_same<bool, T>::value>::type >
  TensorIndex(T boolean) : boolean_(boolean), type_(TensorIndexType::Boolean) {}

  // Case 5: Slice represented in `{start, stop, step}` form,
  // where `start` / `stop` / `step` can be integer or `at::indexing::None`.
  // The tensor form can optionally be provided.
  TensorIndex(
    std::initializer_list<c10::optional<int64_t>> slice,
    at::ArrayRef<Tensor> slice_tensors = {});

  // Case 5: Tensor value
  TensorIndex(Tensor tensor);

  inline bool is_none() const;
  inline bool is_ellipsis() const;

  inline bool is_integer() const;
  inline bool is_integer_with_tensor() const;
  inline int64_t integer() const;

  inline bool is_boolean() const;
  inline bool boolean() const;

  inline bool is_slice() const;
  inline const Slice& slice() const;

  inline bool is_tensor() const;
  inline const Tensor& tensor() const;

 private:
  int64_t integer_;
  bool boolean_;
  Slice slice_;
  Tensor tensor_;
  TensorIndexType type_;
};

// This mirrors `__PySlice_Unpack` in torch/csrc/utils/python_compat.h
inline Slice unpackSlice(
    c10::optional<int64_t> start_index,
    c10::optional<int64_t> stop_index,
    c10::optional<int64_t> step_index,
    const Tensor& start_tensor,
    const Tensor& stop_tensor,
    const Tensor& step_tensor) {
  int64_t start, stop, step;
  if (!step_index.has_value()) {
    step = 1;
  } else {
    step = step_index.value();
    TORCH_CHECK_VALUE(step != 0, "slice step cannot be zero");

    // Here step might be -INDEX_MAX-1; in this case we replace it
    // with -INDEX_MAX.  This doesn't affect the semantics, and it
    // guards against later undefined behaviour resulting from code that
    // does "step = -step" as part of a slice reversal.
    if (step < -INDEX_MAX)
      step = -INDEX_MAX;
  }
  if (!start_index.has_value()) {
    start = step < 0 ? INDEX_MAX : 0;
  } else {
    start = start_index.value();
  }
  if (!stop_index.has_value()) {
    stop = step < 0 ? INDEX_MIN : INDEX_MAX;
  } else {
    stop = stop_index.value();
  }
  return Slice(start, stop, step, start_tensor, stop_tensor, step_tensor);
}

TensorIndex::TensorIndex(c10::nullopt_t) : type_(TensorIndexType::None) {}
TensorIndex::TensorIndex(at::indexing::EllipsisIndexType) : type_(TensorIndexType::Ellipsis) {}
TensorIndex::TensorIndex(const char *str) : TensorIndex(at::indexing::Ellipsis) {
  TORCH_CHECK_VALUE(
    strcmp(str, "...") == 0,
    "Expected \"...\" to represent an ellipsis index, but got \"", str, "\"");
}
TensorIndex::TensorIndex(int64_t integer, Tensor tensor)
    : integer_(integer),
      tensor_(tensor),
      type_(TensorIndexType::Integer) {}
TensorIndex::TensorIndex(int integer) : TensorIndex((int64_t)integer) {}
TensorIndex::TensorIndex(
    std::initializer_list<c10::optional<int64_t>> slice,
    at::ArrayRef<Tensor> slice_tensors)
    : type_(TensorIndexType::Slice) {
  if (slice.size() == 0) {
    slice_ = unpackSlice(c10::nullopt, c10::nullopt, c10::nullopt, {}, {}, {});
  } else if (slice.size() == 2) {
    slice_ = unpackSlice(
      *slice.begin(),
      *(slice.begin() + 1),
      c10::nullopt,
      slice_tensors.size() > 0 ? slice_tensors[0] : Tensor(),
      slice_tensors.size() > 0 ? slice_tensors[1] : Tensor(),
      {});
  } else if (slice.size() == 3) {
    slice_ = unpackSlice(
      *slice.begin(),
      *(slice.begin() + 1),
      *(slice.begin() + 2),
      slice_tensors.size() > 0 ? slice_tensors[0] : Tensor(),
      slice_tensors.size() > 0 ? slice_tensors[1] : Tensor(),
      slice_tensors.size() > 0 ? slice_tensors[2] : Tensor());
  } else {
    TORCH_CHECK_VALUE(
      false,
      "Expected 0 / 2 / 3 elements in the braced-init-list to represent a slice index, but got ",
      slice.size(),
      " element(s)");
  }
}
TensorIndex::TensorIndex(Tensor tensor) : tensor_(tensor), type_(TensorIndexType::Tensor) {}

inline bool TensorIndex::is_none() const {
  return type_ == TensorIndexType::None;
}

inline bool TensorIndex::is_ellipsis() const {
  return type_ == TensorIndexType::Ellipsis;
}

inline bool TensorIndex::is_integer() const {
  return type_ == TensorIndexType::Integer;
}

inline bool TensorIndex::is_integer_with_tensor() const {
  return type_ == TensorIndexType::Integer && tensor_.defined();
}

inline int64_t TensorIndex::integer() const {
  return integer_;
}

inline bool TensorIndex::is_boolean() const {
  return type_ == TensorIndexType::Boolean;
}

inline bool TensorIndex::boolean() const {
  return boolean_;
}

inline bool TensorIndex::is_slice() const {
  return type_ == TensorIndexType::Slice;
}

inline const Slice& TensorIndex::slice() const {
  return slice_;
}

inline bool TensorIndex::is_tensor() const {
  return type_ == TensorIndexType::Tensor;
}

inline const Tensor& TensorIndex::tensor() const {
  return tensor_;
}

CAFFE2_API std::ostream& operator<<(std::ostream& stream, const TensorIndex& tensor_index);
CAFFE2_API std::ostream& operator<<(std::ostream& stream, const std::vector<TensorIndex>& tensor_indices);

inline Tensor applySlice(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t stop,
    int64_t step,
    const Tensor& start_tensor,
    const Tensor& stop_tensor,
    const Tensor& step_tensor,
    bool ensure_view=false) {
  const auto& length = self.size(dim);

  TORCH_CHECK_VALUE(step != 0, "step cannot be zero");
  // TODO: implement negative step
  TORCH_CHECK_VALUE(step >= 0, "negative step not yet supported");

  if (torch::jit::tracer::isTracing() && start_tensor.defined()) {
    torch::jit::tracer::ArgumentStash::stashValue(std::string("start"), 1, start_tensor, torch::jit::IntType::get());
  }
  if (torch::jit::tracer::isTracing() && stop_tensor.defined()) {
    torch::jit::tracer::ArgumentStash::stashValue(std::string("end"), 1, stop_tensor, torch::jit::IntType::get());
  }
  if (torch::jit::tracer::isTracing() && step_tensor.defined()) {
    torch::jit::tracer::ArgumentStash::stashValue(std::string("step"), 1, step_tensor, torch::jit::IntType::get());
  }

  // Skip this optimization if we are tracing, as the trace may be polymorphic
  // over the shape of the `self` tensor, and we still want to record
  // the slice.
  if (!ensure_view && start == 0 && stop == length && step == 1 && !torch::jit::tracer::isTracing()) {
    return self;
  }
  return at::slice(self, dim, start, stop, step);
}

inline Tensor applySelect(const Tensor& self, int64_t dim, int64_t index, const Tensor& index_tensor, int64_t real_dim=0) {
  if (torch::jit::tracer::isTracing() && index_tensor.defined()) {
    torch::jit::tracer::ArgumentStash::stashValue(std::string("index"), 1, index_tensor, torch::jit::IntType::get());
  }

  TORCH_CHECK_INDEX(
    !(index == 0 && dim == 0 && self.dim() == 0),
    "invalid index of a 0-dim tensor. ",
    "Use tensor.item() to convert a 0-dim tensor to a number");

  int64_t size = self.size(dim);
  TORCH_CHECK_INDEX(
    index >= -size && index < size,
    "index ", index, " is out of bounds for dimension ", real_dim, " with size ", size);

  // if the index is negative, do not normalize it because that would fix the index
  // on the current tensor size in the tracer.
  // aten::select also works on negative indices
  return at::select(self, dim, index);
}

// This mirrors `count_specified_dimensions` in torch/csrc/autograd/python_variable_indexing.cpp
inline int64_t count_specified_dimensions(const ArrayRef<TensorIndex>& indices) {
  // Count the number of indexed dimensions (everything but ellipsis and None)
  int64_t count = 0;
  size_t size = indices.size();
  for (size_t i = 0; i < size; i++) {
    auto& obj = indices[i];
    if (obj.is_tensor()) {
      auto& tensor = obj.tensor();
      if (tensor.scalar_type() == kByte || tensor.scalar_type() == kBool) {
        count += tensor.dim();
      } else {
        count++;
      }
    } else if (!obj.is_none() && !obj.is_ellipsis() && !obj.is_boolean()) {
      count++;
    }
  }
  return count;
}

// This mirrors `valueToTensor` in torch/csrc/autograd/python_variable_indexing.cpp
inline Tensor valueToTensor(c10::TensorOptions options, Scalar v) {
  return at::native::scalar_tensor(v, options);
}

// This mirrors `boolToIndexingTensor` in torch/csrc/autograd/python_variable_indexing.cpp
inline Tensor boolToIndexingTensor(const Tensor& self, bool value) {
  // booleans add a dimension of size 1. true indexes this dimension as if 0:, false as empty.
  if (value) {
    return at::native::zeros({1}, {}, self.options().dtype(kLong));
  } else {
    return at::native::empty({0}, {}, self.options().dtype(kLong));
  }
}

inline void _record_tensor_index(const Tensor& tensor, std::vector<Tensor>& outIndices, int64_t& dim) {
  // TODO: check scalarType
  outIndices.resize(dim + 1);
  outIndices[dim] = tensor;
  dim++;
};

inline Tensor handleInteger(const Tensor& self, int64_t& dim, int64_t index, const Tensor& index_tensor, int64_t real_dim) {
  return applySelect(
    self,
    dim,
    index,
    index_tensor,
    real_dim);
}

inline Tensor handleSlice(
  const Tensor& self,
  int64_t& dim,
  int64_t start,
  int64_t stop,
  int64_t step,
  const Tensor& start_tensor,
  const Tensor& stop_tensor,
  const Tensor& step_tensor) {
  Tensor result = applySlice(
    self,
    dim,
    start,
    stop,
    step,
    start_tensor,
    stop_tensor,
    step_tensor);
  dim++;
  return result;
}

inline void handleEllipsis(const Tensor& self, int64_t& dim, int64_t specified_dims) {
  dim += self.dim() - specified_dims;
}

inline Tensor handleNone(const Tensor& self, int64_t& dim) {
  Tensor result = self.unsqueeze(dim);
  dim++;
  return result;
}

inline Tensor handleBoolean(const Tensor& self, bool boolean, std::vector<Tensor>& outIndices, int64_t& dim) {
  Tensor result = self.unsqueeze(dim);
  _record_tensor_index(boolToIndexingTensor(result, boolean), outIndices, dim);
  return result;
}

inline Tensor handleTensor(const Tensor& self, const Tensor& tensor, std::vector<Tensor>& outIndices, int64_t& dim, int64_t real_dim) {
  Tensor result = self;
  auto scalar_type = tensor.scalar_type();
  if (tensor.dim() == 0 && at::isIntegralType(scalar_type, /*includeBool=*/true)) {
    if (scalar_type != at::kByte && scalar_type != at::kBool) {
      result = applySelect(result, dim, tensor.item<int64_t>(), tensor, real_dim);
    } else {
      result = result.unsqueeze(dim);
      if (scalar_type == at::kBool) {
        _record_tensor_index(boolToIndexingTensor(result, tensor.item<bool>() != 0), outIndices, dim);
      } else {
        _record_tensor_index(boolToIndexingTensor(result, tensor.item<uint8_t>() != 0), outIndices, dim);
      }
    }
  } else {
    _record_tensor_index(tensor, outIndices, dim);
  }
  return result;
}

// This mirrors `applySlicing` in torch/csrc/autograd/python_variable_indexing.cpp
inline Tensor applySlicing(const Tensor& self, const ArrayRef<TensorIndex>& indices, std::vector<Tensor>& outIndices) {
  int64_t size = indices.size();
  int64_t dim = 0;
  int64_t specified_dims = count_specified_dimensions(indices);

  TORCH_CHECK_INDEX(specified_dims <= self.dim(), "too many indices for tensor of dimension ", (int)self.dim());

  Tensor result = self;
  for (int64_t i = 0; i < size; i++) {
    auto& obj = indices[i];
    if (obj.is_integer()) {
      result = handleInteger(
        result,
        dim,
        obj.integer(),
        obj.is_integer_with_tensor() ? obj.tensor() : Tensor(),
        i);
    } else if (obj.is_slice()) {
      result = handleSlice(
        result,
        dim,
        obj.slice().start(),
        obj.slice().stop(),
        obj.slice().step(),
        obj.slice().start_tensor(),
        obj.slice().stop_tensor(),
        obj.slice().step_tensor());
    } else if (obj.is_ellipsis()) {
      handleEllipsis(self, dim, specified_dims);
    } else if (obj.is_none()) {
      result = handleNone(result, dim);
    } else if (obj.is_boolean()) {
      result = handleBoolean(result, obj.boolean(), outIndices, dim);
    } else if (obj.is_tensor()) {
      result = handleTensor(result, obj.tensor(), outIndices, dim, i);
    } else {
      TORCH_INTERNAL_ASSERT(false, "Invalid TensorIndex type");
    }
  }
  return result;
}

// This mirrors `typeConvertIndices` in torch/csrc/autograd/python_variable_indexing.cpp
inline std::vector<Tensor> typeConvertIndices(const Tensor& self, const std::vector<Tensor>& indices) {
  std::vector<Tensor> converted_inds(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    const auto &ind = indices[i];
    if (ind.defined()) {
      converted_inds[i] = ind.to(ind.options().device(self.device()));
    } else {
      converted_inds[i] = indices[i];
    }
  }
  return converted_inds;
}

// This mirrors `dispatch_index` in torch/csrc/autograd/python_variable_indexing.cpp
inline Tensor dispatch_index(const Tensor& self, const std::vector<Tensor>& indices) {
  std::vector<Tensor> converted_indices = typeConvertIndices(self, indices);
  OptionalDeviceGuard device_guard(device_of(self));
  return self.index(converted_indices);
}

// This mirrors `dispatch_index_put_` in torch/csrc/autograd/python_variable_indexing.cpp
inline Tensor dispatch_index_put_(Tensor& self, const std::vector<Tensor>& indices, const Tensor& value) {
  std::vector<Tensor> converted_indices = typeConvertIndices(self, indices);
  OptionalDeviceGuard device_guard(device_of(self));
  return self.index_put_(converted_indices, value);
}

// This mirrors `THPVariable_getitem` in torch/csrc/autograd/python_variable_indexing.cpp
inline Tensor get_item(const Tensor& self, const ArrayRef<TensorIndex>& indices) {
  OptionalDeviceGuard device_guard(device_of(self));

  // handle simple types: integers, slices, ellipsis
  if (indices.size() == 1) {
    const TensorIndex& index = indices[0];
    if (index.is_none()) {
      return at::unsqueeze(self, 0);
    } else if (index.is_ellipsis()) {
      return at::alias(self);
    } else if (index.is_integer()) {
      return applySelect(self, 0, index.integer(), index.is_integer_with_tensor() ? index.tensor() : Tensor());
    } else if (index.is_slice()) {
      return applySlice(
        self,
        0,
        index.slice().start(),
        index.slice().stop(),
        index.slice().step(),
        index.slice().start_tensor(),
        index.slice().stop_tensor(),
        index.slice().step_tensor(),
        true);
    }
  }

  std::vector<Tensor> tensorIndices;
  Tensor sliced = applySlicing(self, indices, tensorIndices);
  if (tensorIndices.empty()) {
    if (sliced.is_same(self)) {
      // ensure we return a shallow copy for things like x[...]
      sliced = sliced.alias();
    }
    return sliced;
  }

  // indexing by tensors ("advanced" indexing)
  return dispatch_index(sliced, tensorIndices);
}

// This mirrors `slicePrefix1sSize` in torch/csrc/autograd/python_variable_indexing.cpp
//
// To match numpy semantics:
// As a special case for backwards compatibility,
// strip away unit dimensions from the left of 'src'
inline IntArrayRef slicePrefix1sSize(IntArrayRef sizes) {
  size_t first_non1_src = sizes.size();
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] != 1) {
      first_non1_src = i;
      break;
    }
  }

  return sizes.slice(first_non1_src);
}

// This mirrors `copy_to` in torch/csrc/autograd/python_variable_indexing.cpp
inline void copy_to(Tensor dst, const Tensor& src) {
  Tensor b_src;
  IntArrayRef sliced_src_sizes = slicePrefix1sSize(src.sizes());
  std::tie(b_src) = expand_inplace(dst, src.view(sliced_src_sizes), "setitem");
  dst.copy_(b_src);
}

// This mirrors `THPVariable_setitem` in torch/csrc/autograd/python_variable_indexing.cpp
// for "the assigned value is a Tensor" case
inline void set_item(Tensor& self, const ArrayRef<TensorIndex>& indices, const Tensor& value) {
  OptionalDeviceGuard device_guard(device_of(self));

  // handle simple types: integers, slices, ellipsis, bool
  if (indices.size() == 1) {
    const TensorIndex& index = indices[0];
    if (index.is_boolean() && !index.boolean()) {
      // do nothing for false (technically we should check the size, but we don't have
      // real 0-sized shapes.
      return;
    } else if (index.is_ellipsis()) {
      copy_to(self, value);
      return;
    } else if (index.is_none() || (index.is_boolean() && index.boolean())) {
      copy_to(at::unsqueeze(self, 0), value);
      return;
    } else if (index.is_integer()) {
      copy_to(applySelect(self, 0, index.integer(), index.is_integer_with_tensor() ? index.tensor() : Tensor()), value);
      return;
    } else if (index.is_slice()) {
      copy_to(applySlice(
        self,
        0,
        index.slice().start(),
        index.slice().stop(),
        index.slice().step(),
        index.slice().start_tensor(),
        index.slice().stop_tensor(),
        index.slice().step_tensor()), value);
      return;
    }
  }

  std::vector<Tensor> tensorIndices;
  Tensor sliced = applySlicing(self, indices, tensorIndices);
  if (tensorIndices.empty()) {
    copy_to(sliced, value);
    return;
  }

  IntArrayRef slicedValueSizes = slicePrefix1sSize(value.sizes());
  Tensor valuesSliced;
  if (!value.sizes().equals(slicedValueSizes)) {
    valuesSliced = value.view(slicedValueSizes);
  } else {
    valuesSliced = value;
  }
  dispatch_index_put_(sliced, tensorIndices, valuesSliced);
  return;
}

// This mirrors `set_item` in torch/csrc/autograd/python_variable_indexing.cpp
// for "the assigned value is a Scalar" case
inline void set_item(Tensor& self, const ArrayRef<TensorIndex>& indices, Scalar v) {
  OptionalDeviceGuard device_guard(device_of(self));
  Tensor value;

  // TODO: This qint special case looks very suspicious...
  if (isQIntType(self.scalar_type())) {
    value = valueToTensor(device(kCPU).dtype(kFloat), v);
  } else {
    value = valueToTensor(self.options(), v);
  }

  return set_item(self, indices, value);
}

} // namespace indexing
} // namespace at
