#include <ATen/native/autotune/kernels/convolution.h>

#include <algorithm>
#include <string>

#include <ATen/ATen.h>
#include <ATen/native/autotune/utils/common.h>

namespace autotune {
namespace kernels {

ConvolutionEntryPoint::ConvolutionEntryPoint(const ConvolutionArgs& args)
    : input_sizes_(args.input.sizes()),
      input_strides_(args.input.strides()),
      weight_sizes_(args.weight.sizes()),
      weight_strides_(args.weight.strides()),
      output_sizes_(args.output_sizes),
      itemsize_(args.input.itemsize()) {
  // Autotuning is only prototyped on a subset of Conv2D.
  bool supported =
      (args.input.options().backend() == at::Backend::CPU &&
       args.input.scalar_type() == at::kFloat &&
       args.weight.scalar_type() == at::kFloat &&
       args.weight.ndimension() == 4 &&
       std::all_of(
           args.dilation.begin(),
           args.dilation.end(),
           [](int64_t x) { return x == 1; }) &&
       !args.is_transposed && !args.is_depthwise && args.groups == 1);

  fallback_ = !supported;
  if (supported)
    compute_hash({input_sizes_,
                  input_strides_,
                  weight_sizes_,
                  weight_strides_,
                  output_sizes_,
                  {itemsize_}});
}

bool ConvolutionEntryPoint::fallback() {
  return fallback_;
}

Task ConvolutionEntryPoint::task() {
  return Task::kConv2D;
}

selection::KernelEntryPoint::cost_estimates ConvolutionEntryPoint::costs() {
  int64_t output_numel = 1;
  for (auto i : output_sizes_)
    output_numel *= i;

  // This currently assumes Conv2D, which is enforced when determining whether
  // to fallback.
  auto batch_size = input_sizes_[0];
  auto channels_out = weight_sizes_[0];
  auto channels_in = weight_sizes_[1];

  // Height * Width
  auto kernel_hw = weight_sizes_[2] * weight_sizes_[3];
  auto output_hw = output_sizes_[2] * output_sizes_[3];

  auto read_bytes =
      (util::bytes_span(input_sizes_, input_strides_, itemsize_) +
       util::bytes_span(weight_sizes_, weight_strides_, itemsize_));
  auto write_bytes = output_numel * itemsize_;

  // Naive rooflines.
  double memory_roofline =
      ((double)read_bytes / system::memory_bandwidth::sequential_read +
       (double)write_bytes / system::memory_bandwidth::sequential_write);

  double compute_roofline =
      (double)(batch_size * channels_out * channels_out * kernel_hw * output_hw) /
      system::cpu_hz;

  double roofline =
      std::max({memory_roofline, compute_roofline}) + convolution_overhead;

  // For now use the same roofline for all implementations.
  // This will be refined later.
  return {{Implementation::kConv2D_Native, roofline},
          {Implementation::kConv2D_NNPack, roofline},
          {Implementation::kConv2D_MKL, roofline}};
}

selection::KernelEntryPoint::supported_implementations ConvolutionEntryPoint::
    implementations() {
  return {Implementation::kConv2D_Native,
          Implementation::kConv2D_NNPack,
          Implementation::kConv2D_MKL};
}

std::string ConvolutionEntryPoint::repr() {
  // Ignore strides for now, and output size is
  // a function of input size and kernel size since
  // we are ignoring convolution padding/strides/etc.
  // for now.
  return autotune::utils::string_format(
      "Convolution: input_size = (%d,%d,%d,%d), "
      "weight_size = (%d,%d,%d,%d)",
      input_sizes_[0],
      input_sizes_[1],
      input_sizes_[2],
      input_sizes_[3],
      weight_sizes_[0],
      weight_sizes_[1],
      weight_sizes_[2],
      weight_sizes_[3]);
}

} // namespace kernels
} // namespace autotune
