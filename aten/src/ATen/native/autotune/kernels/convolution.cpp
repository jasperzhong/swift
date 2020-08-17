#include <ATen/native/autotune/kernels/convolution.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
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
  TORCH_INTERNAL_ASSERT(num_threads_ > 0);

  auto tensor_supported = [](const at::Tensor& x) {
    return (
        x.options().backend() == at::Backend::CPU &&
        x.scalar_type() == at::kFloat && x.is_contiguous() &&
        x.ndimension() == 4);
  };

  // Autotuning is only prototyped on a subset of Conv2D.
  bool supported =
      (tensor_supported(args.input) && tensor_supported(args.weight) &&
       std::all_of(
           args.dilation.begin(),
           args.dilation.end(),
           [](int64_t x) { return x == 1; }) &&
       !args.is_transposed && !args.is_depthwise && args.groups == 1);

  // It is common for the very end of conv-nets to be very narrow and very
  // deep. (e.g. batch x 2 x 2 x 512) This is pathalogical for the spectral
  // (FFT and Winograd) kernels that NNPack uses for those cases. By contrast
  // we know that the GEMM approach of Native and MKL will provide at least
  // reasonable performance, so there's no point in even attempting NNPack
  // for this subset.
  skip_nnpack_ =
      (input_sizes_[2] <= nnpack_spectal_tile_threshold &&
       input_sizes_[3] <= nnpack_spectal_tile_threshold &&
       weight_sizes_[2] > 1 && weight_sizes_[3] > 1);

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

double shard_work(double work, double effective_thread_count) {
  // Computations are computed in double rather than int to allow
  // the effective thread count to be fractional.
  auto grain_size = (double)(at::internal::GRAIN_SIZE);
  auto work_per_pass = grain_size * effective_thread_count;
  auto complete_passes = std::floor(work / work_per_pass);
  auto remainder = work - complete_passes * work_per_pass;
  return complete_passes * grain_size + remainder;
}

double ConvolutionEntryPoint::cost(const Conv2DPerfParams& impl_params) {
  TORCH_INTERNAL_ASSERT(
      impl_params.impl == api::Implementation::kConv2D_Native ||
      impl_params.impl == api::Implementation::kConv2D_NNPack ||
      impl_params.impl == api::Implementation::kConv2D_MKL);

  // This currently assumes Conv2D, which is enforced when determining whether
  // to fallback.
  auto batch_size = input_sizes_[0];
  auto image_h = input_sizes_[2];
  auto image_w = input_sizes_[3];

  auto channels_out = weight_sizes_[0];
  auto channels_in = weight_sizes_[1];
  auto kernel_h = weight_sizes_[2];
  auto kernel_w = weight_sizes_[3];

  // Currently we do not consider padding, so this is a temporary
  // hack to bridge the interim.
  auto out_h = std::max({image_h - kernel_h + 1, (int64_t)1});
  auto out_w = std::max({image_w - kernel_w + 1, (int64_t)1});

  auto kernel_hw = kernel_h * kernel_w;
  auto image_hw = image_h * image_w;

  auto is_1x1 = (kernel_hw == 1);
  auto is_3x3 = (kernel_hw == 9);
  auto fft_candidate = (kernel_h > 1 || kernel_w > 1) && kernel_h <= 16 && kernel_w <= 16;

  int64_t gemm_effective_inner_size =
      std::max({system::cpu_vector_size, channels_in * kernel_hw});
  int64_t gemm_effective_channels_out =
      std::max({impl_params.min_effective_c_out, channels_out});

  int64_t max_block_size = std::min(
      {out_h * out_w, gemm_effective_inner_size, gemm_effective_channels_out});
  double blas_factor = 1.0 / std::pow((double)(max_block_size), 0.2);

  int64_t im2col = is_1x1
      ? itemsize_ * batch_size * image_hw * kernel_hw * channels_in
      : 0; // Im2Col is a no-op for 1x1

  // FFT runs over a minimum 8x8 patch, and Winograd is written directly in
  // AVX2 assembly. As a result, both will incur significant waste when
  // out_h * out_w is very small.
  auto spectral_out_hw = std::max({out_h * out_w, (int64_t)64});

  // This doesn't cover all of the nuance, such as the fact that
  // NCHW <-> NHWC is a no-op if C=1 or the fact that read and write bandwidth
  // is generally not symetric. It's just to give a general estimate for the
  // number of bytes which potentially need to be moved.
  int64_t layout_transform = (
      // Input Tensor
      itemsize_ * batch_size * channels_in * image_hw +

      // Output Tensor
      itemsize_ * batch_size * channels_out * out_h * out_w +

      // Weight Tensor. (bias should be negligible.)
      itemsize_ * channels_in * channels_out * kernel_hw);

  auto gemm_work = [=]() {
    auto left_size = batch_size * out_h * out_w * gemm_effective_inner_size;
    return (double)(left_size * gemm_effective_channels_out) * blas_factor;
  };

  double im2col_portion = 0;
  double compute_portion;
  if (is_1x1) {
    // For 1x1, direct GEMM is always fastest.
    compute_portion = gemm_work();
  } else if (impl_params.supports_3x3_winograd && is_3x3) {
    // Winograd requires two ops per element instead of kernel_hw.
    compute_portion =
        (double)(batch_size * channels_out * channels_in * spectral_out_hw * 2);
  } else if (impl_params.supports_fft || fft_candidate) {
    // FFT requires four ops per element instead of kernel_hw.
    compute_portion =
        (double)(batch_size * channels_out * channels_in * spectral_out_hw * 4);
  } else {
    compute_portion = gemm_work();
    im2col_portion = (double)im2col;
  }

  double effective_num_threads =
      std::pow((double)(num_threads_), impl_params.thread_scaling);
  auto shard = [effective_num_threads](double work) {
    return shard_work(work, effective_num_threads);
  };

  auto cost = (
    // Compute has been formulated assuming SIMD, including padding up to the
    // vectorization width. Thus, it is fair to unconditionally divide here.
    shard(compute_portion) / (double)(system::cpu_vector_size) +

    shard((double)layout_transform) * impl_params.layout_factor +
    shard(im2col_portion) * impl_params.im2col_factor
  ) * impl_params.overall_scale + (double)impl_params.overhead;

  return cost / system::cpu_hz;
}

selection::KernelEntryPoint::cost_estimates ConvolutionEntryPoint::costs() {
  selection::KernelEntryPoint::cost_estimates out{
    {api::Implementation::kConv2D_Native, cost(conv2d_native_perf)},
    {api::Implementation::kConv2D_MKL, cost(conv2d_mkl_perf)}
  };
  if (!skip_nnpack_)
    out.push_back({api::Implementation::kConv2D_NNPack, cost(conv2d_nnpack_perf)});
  return out;
}

selection::KernelEntryPoint::supported_implementations ConvolutionEntryPoint::
    implementations() {
  selection::KernelEntryPoint::supported_implementations out {
    api::Implementation::kConv2D_Native,
    api::Implementation::kConv2D_MKL};
  if (!skip_nnpack_)
    out.push_back(api::Implementation::kConv2D_NNPack);
  return out;
}

std::string ConvolutionEntryPoint::repr() {
  // Output size is a function of input size and kernel size since
  // we are ignoring convolution padding/strides/etc. for now.
  return autotune::utils::string_format(
      "Convolution: input_size = (%s), "
      "weight_size = (%s), %d threads",
      logging::to_string(input_sizes_).c_str(),
      logging::to_string(weight_sizes_).c_str(),
      num_threads_);
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
       /*groups=        */ 1,
       /*num_threads=   */ at::get_num_threads()});

  auto choice = dispatch.choice();
  auto output = convolution_2D(x, weight, choice);

  // This is only necessary because of the overload which
  // bypasses dispatch alltogether.
  if (choice == api::Implementation::kConv2D_Native or
      choice == api::Implementation::kConv2D_NNPack or
      choice == api::Implementation::kConv2D_MKL) {
    dispatch.finish();
  }

  return output;
}

} // namespace autotune
