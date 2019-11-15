#pragma once

#include <c10/util/Optional.h>
#include <ATen/core/TensorBody.h>

namespace at {
namespace indexing {

const int64_t INDEX_MAX = std::numeric_limits<int64_t>::max();
const int64_t INDEX_MIN = std::numeric_limits<int64_t>::min();

enum class TensorIndexType { None, Ellipsis, Integer, Boolean, Slice, Tensor };

constexpr c10::nullopt_t None{c10::nullopt_t::init()};

struct CAFFE2_API EllipsisIndexType { EllipsisIndexType() {} };
CAFFE2_API extern const EllipsisIndexType Ellipsis;

struct CAFFE2_API Slice {
 public:
  Slice();
  Slice(int64_t start, int64_t stop, int64_t step);

  const int64_t& start() const;
  const int64_t& stop() const;
  const int64_t& step() const;

 private:
  int64_t start_;
  int64_t stop_;
  int64_t step_;
};

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
struct CAFFE2_API TensorIndex {
  // Case 1: `at::indexing::None`
  TensorIndex(c10::nullopt_t);

  // Case 2: "..." / `at::indexing::Ellipsis`
  TensorIndex(at::indexing::EllipsisIndexType);
  TensorIndex(const char *str);

  // Case 3: Integer value
  TensorIndex(int64_t integer);
  TensorIndex(int integer);

  // Case 4: Boolean value
  TensorIndex(bool boolean);

  // Case 5: Slice represented in `{start, stop, step}` form,
  // where `start` / `stop` / `step` can be integer or `at::indexing::None`
  TensorIndex(std::initializer_list<c10::optional<int64_t>> init_list);

  // Case 5: Tensor value
  TensorIndex(Tensor tensor);

  bool is_none() const;
  bool is_ellipsis() const;

  bool is_integer() const;
  int64_t integer() const;

  bool is_boolean() const;
  bool boolean() const;

  bool is_slice() const;
  const Slice& slice() const;

  bool is_tensor() const;
  const Tensor& tensor() const;

 private:
  int64_t integer_;
  bool boolean_;
  Slice slice_;
  Tensor tensor_;
  TensorIndexType type_;
};

struct CAFFE2_API TensorMultiDimIndexingMeta {
  // yf225 TODO: hmm do we need to handle other operators as well? (like `+=`?) Try it out in Python and see what the expected behavior is!
  /* yf225 TODO
  1. how would it work in Python if `b = a[advanced indexing with tensor], b = some scalar value`? and `b = a[advanced indexing with tensor], b.zero_()`? compared it to C++ API behavior!
  2. also how would a[adv indexing][adv indexing] = some scalar value work in Python and in C++? try it out in both languages
  */
  // yf225 TODO: what happens if during `a(indices) = func()`, the RHS throws an exception? How do we clean up the thread-local TensorMultiDimIndexingMeta in this case?
  static bool has_indexing_history();
  static void save_indexing_history(Tensor pre_indexing_tensor, Tensor post_indexing_tensor, std::vector<TensorIndex> indices);
  static void clear_indexing_history();
  static bool has_indexing_history_for(const Tensor& tensor);
  static void assign_value_using_indexing(Tensor const & rhs);
  static void assign_value_using_indexing(Tensor && rhs);
  static void assign_value_using_indexing(Scalar v);
};

CAFFE2_API std::ostream& operator<<(std::ostream& stream, const TensorIndex& tensor_index);
CAFFE2_API std::ostream& operator<<(std::ostream& stream, const std::vector<TensorIndex>& tensor_indices);

} // namespace indexing
} // namespace at
