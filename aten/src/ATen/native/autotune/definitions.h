#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <c10/util/ArrayRef.h>

namespace autotune {

enum class DispatchGroup {
  kConv2D,

  kNotApplicable,
};

enum class DispatchChoice {
  kConv2D_Native = 0,
  kConv2D_MKL,

  kFallback,
  kUnsupported,

  TOTAL_COUNT, // Must be the final element.
};

constexpr size_t NumDispatchChoice = (size_t)DispatchChoice::TOTAL_COUNT;

// Currently the dispatch key and cost function are computed together,
// which is a bit wasteful but significantly simplifies the prototype.
class EntryPoint {
 public:
  struct ImplementationPrior;
  using impl_priors = std::vector<ImplementationPrior>;

  // Key collisions within a dispatch group are acceptable;
  // collisions across groups are not. As a result, the group
  // must be reserved as part of the map key. Consequently, we
  // may as well use the other four bytes to reduce collisions
  // since std::pair<DispatchGroup, uint64_t> would pad to size 16.
  using map_key = std::tuple<DispatchGroup, uint32_t, uint64_t>;
  static_assert(sizeof(map_key) == 16);

  EntryPoint(
      DispatchGroup group,
      impl_priors priors,
      std::initializer_list<c10::IntArrayRef> hash_criteria)
      : group_(group), priors_(priors) {
    for (auto i : hash_criteria)
      hash(i);
  };

  static EntryPoint Fallback() {
    return {
        DispatchGroup::kNotApplicable, {{DispatchChoice::kFallback, 0.0}}, {}};
  };

  static EntryPoint Unsupported() {
    return {
        DispatchGroup::kNotApplicable, {{DispatchChoice::kUnsupported, 0.0}}, {}};
  };

  map_key key() {
    return {
        group_,
        static_cast<uint32_t>(hash_x0_ % std::numeric_limits<uint32_t>::max()),
        hash_x1_};
  };

  impl_priors value() {
    return priors_;
  };

  struct ImplementationPrior {
    DispatchChoice impl;
    double cost;
  };

 private:
  DispatchGroup group_;
  impl_priors priors_;
  uint64_t hash_x0_{0};
  uint64_t hash_x1_{1};

  void hash(c10::IntArrayRef x) {
    for (auto xi : x) {
      // https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
      hash_x0_ ^= std::hash<int64_t>{}(xi) + 0x9e3779b9 + (hash_x0_ << 6) +
          (hash_x0_ >> 2);
      hash_x1_ ^= std::hash<int64_t>{}(xi) + 0x9e3779b9 + (hash_x1_ << 6) +
          (hash_x1_ >> 2);
    }
  };
};

// https://stackoverflow.com/a/26221725
template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
  size_t size =
      snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
  if (size <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(
      buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

} // namespace autotune
