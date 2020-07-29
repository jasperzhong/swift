#pragma once

#include <cstddef>

#include <c10/util/ArrayRef.h>

namespace autotune {
namespace kernels {
enum class Task {
  kConv2D,

  kNotApplicable,
};

enum class Implementation {
  kConv2D_Native = 0,
  kConv2D_NNPack,
  kConv2D_MKL,

  // Autotuning has not been enabled in at::globalContext.
  kDisabled,

  // The kernel entry point has elected not to use autotuning.
  kFallback,

  // An implementation cannot be selected for an unknown reason.
  kUnsupported,

  TOTAL_COUNT, // Must be the final element.
};
constexpr size_t NumImplementations = (size_t)Implementation::TOTAL_COUNT;

} // namespace kernels

namespace system {
// Placeholder. These should be determined on a system by system basis.
namespace memory_bandwidth {
static int64_t sequential_read = 7'000'000'000;
static int64_t random_read = 1'600'000'000;
static int64_t sequential_write = 5'000'000'000;
static int64_t random_write = 400'000'000;
} // namespace memory_bandwidth

static double cpu_hz = 2'394'444'000;
static int64_t cpu_vector_size = 16; // Broadwell, fp32
static int64_t cache_line_size = 64;
} // namespace system

namespace util {

// Compute how many physical bytes a Tensor spans.
// Used to estimate the amount of time to fetch the contents from memory.
static size_t bytes_span(
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides,
    size_t itemsize) {
  auto dim = sizes.size();
  if (!dim)
    return 0;

  size_t output = 1;
  for (int i = 0; i < dim; i++) {
    auto size = sizes[i];
    if (size > 1)
      output += (size - 1) * strides[i];
  }

  return output * itemsize;
}
} // namespace util

} // namespace autotune
