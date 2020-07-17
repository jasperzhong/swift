#include <ATen/native/autotune/cost.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>
#include <ATen/native/autotune/definitions.h>


namespace cost {

size_t bytes_span(c10::IntArrayRef sizes, c10::IntArrayRef strides, int64_t itemsize) {
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

autotune::cost_priors conv_priors(
    c10::IntArrayRef input_sizes,
    c10::IntArrayRef input_strides,
    c10::IntArrayRef weight_sizes,
    c10::IntArrayRef weight_strides,
    c10::IntArrayRef output_sizes,
    int64_t itemsize) {
  auto read_bytes =
      (bytes_span(input_sizes, input_strides, itemsize) +
       bytes_span(weight_sizes, weight_strides, itemsize));

  int64_t output_numel = 1;
  for (auto i : output_sizes)
    output_numel *= i;

  auto N = input_sizes[0];
  auto C_out = output_sizes[0];
  auto C_in = output_sizes[1];
  auto kernel_hw = weight_sizes[2] * weight_sizes[3];
  auto output_hw = output_sizes[2] * output_sizes[3];

  // Naive rooflines.
  double memory_roofline = (
      (double)read_bytes / main_memory::bandwidth::sequential_read +
      (double)output_numel * itemsize / main_memory::bandwidth::sequential_write);

  double compute_roofline =
      (double)(N * C_in * C_out * kernel_hw * output_hw) / cpu_hz;

  double roofline = std::max(memory_roofline, compute_roofline);

  // For now use the same roofline for both implementations.
  // This will be refined later.
  return {
      {autotune::DispatchChoice::kConv2D_Native, roofline},
      {autotune::DispatchChoice::kConv2D_MKL,    roofline}
  };
}
















// https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
void hash_combine(size_t & seed, c10::IntArrayRef v){
    for (auto vi : v) seed ^= std::hash<int64_t>{}(vi) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}


namespace conv2d {

size_t Roofline::key() {
    size_t output = 0;
    hash_combine(output, input_sizes_);
    hash_combine(output, input_strides_);
    hash_combine(output, weight_sizes_);
    hash_combine(output, weight_strides_);
    hash_combine(output, output_sizes_);
    return output;
}




std::string Roofline::repr() {
    return autotune::string_format(
        "Input sizes  (strides):   %d, %d, %d, %d     (%d, %d, %d, %d)\n"
        "Weight sizes (strides):   %d, %d, %d, %d     (%d, %d, %d, %d)\n"
        "Output sizes:             %d, %d, %d, %d",
        input_sizes_[0], input_sizes_[1], input_sizes_[2], input_sizes_[3],
        input_strides_[0], input_strides_[1], input_strides_[2], input_strides_[3],
        weight_sizes_[0], weight_sizes_[1], weight_sizes_[2], weight_sizes_[3],
        weight_strides_[0], weight_strides_[1], weight_strides_[2], weight_strides_[3],
        output_sizes_[0], output_sizes_[1], output_sizes_[2], output_sizes_[3]
    );
}


std::vector<double> Roofline::compute() {
  auto read_bytes =
      (bytes_span(input_sizes_, input_strides_, itemsize_) +
       bytes_span(weight_sizes_, weight_strides_, itemsize_));

  int64_t output_numel = 1;
  for (auto i : output_sizes_)
    output_numel *= i;

  auto elements_per_cache_line = cache_line_size / itemsize_;
  auto cache_lines_fetched =
      (read_bytes + elements_per_cache_line - 1) / elements_per_cache_line;
  double read_stall_time =
      (double)(cache_lines_fetched)*main_memory::approx_latency;

  double memory = ((double)read_bytes / main_memory::bandwidth::sequential_read +
             (double)output_numel * itemsize_ / main_memory::bandwidth::sequential_write);
  double memory_with_stalls = memory + read_stall_time;

  auto N = input_sizes_[0];
  auto C_out = output_sizes_[0];
  auto C_in = output_sizes_[1];
  auto kernel_hw = weight_sizes_[2] * weight_sizes_[3];
  auto output_hw = output_sizes_[2] * output_sizes_[3];

  auto C_in_vectorized = (C_in + cpu_vector_size - 1) / cpu_vector_size;
  auto kernel_hw_vectorized =
      (kernel_hw + cpu_vector_size - 1) / cpu_vector_size;

  double compute_naive =
      (double)(N * C_in * C_out * kernel_hw * output_hw) / cpu_hz;
  double compute_C_in_vectorized =
      (double)(N * C_in_vectorized * C_out * kernel_hw * output_hw) / cpu_hz;
  double compute_kernel_hw_vectorized =
      (double)(N * C_in * C_out * kernel_hw_vectorized * output_hw) / cpu_hz;

  return {
      // Native
      std::max(memory_with_stalls, compute_naive),

      // MKL DNN
      std::max(memory, std::min(compute_C_in_vectorized, compute_kernel_hw_vectorized))};
}


} // namespace conv2d
} // namespace cost
