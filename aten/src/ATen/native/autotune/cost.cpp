#include <ATen/native/autotune/cost.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/native/autotune/definitions.h>
#include <c10/util/ArrayRef.h>

namespace cost {

size_t bytes_span(
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides,
    int64_t itemsize) {
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

autotune::EntryPoint conv_priors(
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
  double memory_roofline =
      ((double)read_bytes / main_memory::bandwidth::sequential_read +
       (double)output_numel * itemsize /
           main_memory::bandwidth::sequential_write);

  double compute_roofline =
      (double)(N * C_in * C_out * kernel_hw * output_hw) / cpu_hz;

  double roofline = std::max(memory_roofline, compute_roofline);

  // For now use the same roofline for both implementations.
  // This will be refined later.
  return {
    /*group=        */autotune::DispatchGroup::kConv2D,
    /*priors=       */{{autotune::DispatchChoice::kConv2D_Native, roofline},
                       {autotune::DispatchChoice::kConv2D_MKL, roofline}},
    /*hash_criteria=*/{input_sizes, input_strides, weight_sizes, weight_strides, output_sizes}};
}

autotune::EntryPoint dispatch_conv(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::IntArrayRef output_sizes,
    c10::IntArrayRef dilation,
    bool is_transposed,
    bool is_depthwise,
    int64_t groups) {

  if (!at::globalContext().userEnabledCost())
    return autotune::EntryPoint::Fallback();

  // Autotuning is only prototyped on a subset of Conv2D.
  bool supported = (
    input.options().backend() == at::Backend::CPU &&
    input.scalar_type() == at::kFloat &&
    weight.scalar_type() == at::kFloat &&
    weight.ndimension() == 4 &&
    !is_transposed &&
    !is_depthwise &&
    groups == 1 &&
    std::all_of(dilation.begin(), dilation.end(), [](int64_t x){ return x == 1; })
  );

  if (!supported)
    return autotune::EntryPoint::Unsupported();

  return conv_priors(
        input.sizes(),
        input.strides(),
        weight.sizes(),
        weight.strides(),
        output_sizes,
        input.itemsize());
}

} // namespace cost
