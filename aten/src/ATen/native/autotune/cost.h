#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>

#include <ATen/native/autotune/dispatch.h>


namespace cost {

// Placeholder. These should be determined on a system by system basis.
namespace main_memory {
namespace bandwidth {
static int64_t sequential_read = 7'000'000'000;
static int64_t random_read = 1'600'000'000;
static int64_t sequential_write = 5'000'000'000;
static int64_t random_write = 400'000'000;
} // namespace bandwidth

static double approx_latency = 10.0 * 1e-9;
} // namespace main_memory
static double cpu_hz = 2'394'444'000;
static int64_t cpu_vector_size = 16; // Broadwell, fp32
static int64_t cache_line_size = 64;


autotune::cost_priors conv_priors(
  c10::IntArrayRef input_sizes,
  c10::IntArrayRef input_strides,
  c10::IntArrayRef weight_sizes,
  c10::IntArrayRef weight_strides,
  c10::IntArrayRef output_sizes,
  int64_t itemsize
);














namespace conv2d {

class Roofline {
 public:
  Roofline(
      const at::Tensor& input,
      const at::Tensor& weight,
      c10::IntArrayRef output_sizes)
      : input_sizes_(input.sizes()), input_strides_(input.strides()),
        weight_sizes_(weight.sizes()), weight_strides_(weight.strides()),
        output_sizes_(output_sizes), itemsize_(input.itemsize()) {};

  size_t key();
  std::string repr();
  std::vector<double> compute();

 private:
  c10::IntArrayRef input_sizes_;
  c10::IntArrayRef input_strides_;
  c10::IntArrayRef weight_sizes_;
  c10::IntArrayRef weight_strides_;
  c10::IntArrayRef output_sizes_;
  int64_t itemsize_;
};

} // namespace conv2d
} // namespace cost
