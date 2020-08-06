#include <ATen/native/autotune/kernels/convolution.h>

#include <algorithm>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/autotune/api.h>
#include <ATen/native/autotune/dispatch/common.h>
#include <ATen/native/autotune/dispatch/core.h>
#include <ATen/native/autotune/utils/common.h>
#include <ATen/native/autotune/utils/logging.h>
#include <c10/util/Exception.h>

namespace autotune {
namespace kernels {

ConvolutionEntryPoint::ConvolutionEntryPoint(const ConvolutionArgs& args)
    : input_sizes_(args.input.sizes()),
      weight_sizes_(args.weight.sizes()),
      output_sizes_(args.output_sizes),
      itemsize_(args.input.itemsize()),
      num_threads_(args.num_threads) {
  // Autotuning is only prototyped on a subset of Conv2D.
  bool supported =
      (args.input.options().backend() == at::Backend::CPU &&
       args.input.scalar_type() == at::kFloat &&
       args.weight.scalar_type() == at::kFloat && args.input.is_contiguous() &&
       args.weight.is_contiguous() && args.weight.ndimension() == 4 &&
       std::all_of(
           args.dilation.begin(),
           args.dilation.end(),
           [](int64_t x) { return x == 1; }) &&
       !args.is_transposed && !args.is_depthwise && args.groups == 1);

  fallback_ = !supported;
  if (supported)
    // Only bother to set up the feature vector if it might be used.
    declare_features({input_sizes_,
                      weight_sizes_,
                      output_sizes_,
                      {itemsize_, num_threads_}});
}

bool ConvolutionEntryPoint::fallback() {
  return fallback_;
}

api::Task ConvolutionEntryPoint::task() {
  return api::Task::kConv2D;
}

int64_t product(c10::IntArrayRef x) {
  return std::accumulate(x.begin(), x.end(), 1, std::multiplies<int>());
}

selection::KernelEntryPoint::cost_estimates ConvolutionEntryPoint::costs() {
  int64_t output_numel = product(output_sizes_);

  // This currently assumes Conv2D, which is enforced when determining whether
  // to fallback.
  auto batch_size = input_sizes_[0];
  auto channels_out = weight_sizes_[0];
  auto channels_in = weight_sizes_[1];

  // Height * Width
  auto kernel_hw = weight_sizes_[2] * weight_sizes_[3];
  auto output_hw = output_sizes_[2] * output_sizes_[3];

  auto read_numel = product(input_sizes_) + product(weight_sizes_);
  auto read_bytes = read_numel * itemsize_;
  auto write_bytes = product(output_sizes_) * itemsize_;

  // Naive rooflines.
  double memory_roofline =
      ((double)read_bytes / system::memory_bandwidth::sequential_read +
       (double)write_bytes / system::memory_bandwidth::sequential_write);

  double compute_roofline =
      (double)(batch_size * channels_out * channels_out * kernel_hw * output_hw) /
      system::cpu_hz;

  double roofline =
      std::max({memory_roofline, compute_roofline}) / (double)num_threads_ +
      convolution_overhead;

  // For now use the same roofline for all implementations.
  // This will be refined later.
  return {{api::Implementation::kConv2D_Native, roofline},
          {api::Implementation::kConv2D_NNPack, roofline},
          {api::Implementation::kConv2D_MKL, roofline}};
}

selection::KernelEntryPoint::supported_implementations ConvolutionEntryPoint::
    implementations() {
  return {api::Implementation::kConv2D_Native,
          api::Implementation::kConv2D_NNPack,
          api::Implementation::kConv2D_MKL};
}

std::string ConvolutionEntryPoint::repr() {
  // Output size is a function of input size and kernel size since
  // we are ignoring convolution padding/strides/etc. for now.
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

// Temporary. Evenentually this should be rolled into at::_convolution.
static std::vector<int64_t> conv2d_stride{1, 1};
static std::vector<int64_t> conv2d_dilation{1, 1};
static std::vector<int64_t> conv2d_padding{0, 0};
static std::vector<int64_t> conv2d_output_padding{0, 0};

at::Tensor convolution_2D(
    at::Tensor& x,
    at::Tensor& weight,
    api::Implementation choice) {
  auto bias = at::ones({weight.sizes()[0]});

  switch (choice) {
    case api::Implementation::kDisabled:
      return at::convolution(
          x,
          weight,
          bias,
          conv2d_stride,
          conv2d_padding,
          conv2d_dilation,
          /*transposed=*/false,
          conv2d_output_padding,
          /*groups=*/1);

    case api::Implementation::kConv2D_Native:
      return at::thnn_conv2d(
          x,
          weight,
          weight.sizes().slice(2),
          bias,
          conv2d_stride,
          conv2d_padding);

    case api::Implementation::kConv2D_NNPack:
      TORCH_INTERNAL_ASSERT(at::_nnpack_available());
      return at::_nnpack_spatial_convolution(
          x, weight, bias, conv2d_padding, conv2d_stride);

    case api::Implementation::kConv2D_MKL:
      return at::mkldnn_convolution(
          x.contiguous(),
          weight.contiguous(),
          bias.contiguous(),
          conv2d_padding,
          conv2d_stride,
          conv2d_dilation,
          /*groups=*/1);
    case api::Implementation::kFallback:
    case api::Implementation::kUnsupported:
    default:
      AT_ERROR("Currently unsupported: ", logging::to_string(choice));
  }
}

at::Tensor convolution_2D(at::Tensor& x, at::Tensor& weight) {
  // This will also initialize NNPack.
  TORCH_INTERNAL_ASSERT(at::_nnpack_available());

  auto output_sizes = at::native::conv_output_size(
      x.sizes(),
      weight.sizes(),
      conv2d_padding,
      conv2d_stride,
      conv2d_dilation);

  auto dispatch = selection::DispatchConvolution( //
      {/*input=         */ x,
       /*weight=        */ weight,
       /*output_sizes=  */ output_sizes,
       /*dilation=      */ conv2d_dilation,
       /*is_transposed= */ false,
       /*is_depthwise=  */ false,
       /*groups=        */ 1});

  auto choice = dispatch.choice();
  auto output = convolution_2D(x, weight, choice);
  if (choice == api::Implementation::kConv2D_Native or
      choice == api::Implementation::kConv2D_Native or
      choice == api::Implementation::kConv2D_NNPack) {
    dispatch.finish();
  }

  return output;
}

} // namespace autotune
