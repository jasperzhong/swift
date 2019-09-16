#include <c10/util/Exception.h>

namespace at {
namespace idx {

enum class SpecialIndexingType { None, Ellipsis };

using None = SpecialIndexingType::None;
using Ellipsis = SpecialIndexingType::Ellipsis;

struct Slice {
 public:
  /* Supported slice syntax:
  {}
  {0, 10}
  {0, None}
  {0, 10, 2}
  {0, None, 2}
  */
  Slice() : start_(0), stop_(std::numeric_limits<int64_t>::max()), step_(1) {}
  Slice(int64_t start, at::idx::None stop) : start_(start), stop_(std::numeric_limits<int64_t>::max()), step_(1) {}
  Slice(int64_t start, int64_t stop) : start_(start), stop_(stop), step_(1) {}
  Slice(int64_t start, at::idx::None stop, int64_t step)
      : start_(start), stop_(std::numeric_limits<int64_t>::max()), step_(step) {}
  Slice(int64_t start, int64_t stop, int64_t step) : start_(start), stop_(stop), step_(step) {}

  const int64_t& start() {
    return start_;
  }

  const int64_t& stop() {
    return stop_;
  }

  const int64_t& step() {
    return step_;
  }

 private:
  int64_t start_;
  int64_t stop_;
  int64_t step_;
};

// yf225 TODO: call this from python_variable_indexing.cpp
inline Tensor applySelect(const Tensor& self, int64_t dim, int64_t index, int64_t real_dim=0) {
  if (index == 0 && dim == 0 && self.dim() == 0) {
    throw IndexError(
        "invalid index of a 0-dim tensor. "
        "Use tensor.item() to convert a 0-dim tensor to a Python number"); // yf225 TODO: change error message
  }
  int64_t size = self.size(dim);
  if (index < -size || index >= size) {
    throw IndexError("index %lld is out of bounds for dimension %lld with size %lld",
      index, real_dim, size);
  }
  // if the index is negative, do not normalize it because that would fix the index
  // on the current tensor size in the tracer.
  // aten::select also works on negative indices
  return self.select(dim, index);
}

// yf225 TODO: call this from python_variable_indexing.cpp (with refactoring the callsite)
// yf225 TODO: fix this
inline Tensor applySlice(const Tensor& self, int64_t dim, const Slice& slice, bool ensure_view=false) {
  auto& start = slice.start();
  auto& stop = slice.stop();
  auto& step = slice.step();
  if (step == 0) {
    throw ValueError("step cannot be zero");
  }
  if (step < 0) {
    // TODO: implement negative step
    throw ValueError("negative step not yet supported");
  }
  return self.slice(dim, start, stop, step);
}

} // namespace idx
} // namespace at
