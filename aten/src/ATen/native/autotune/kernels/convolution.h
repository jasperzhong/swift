#pragma once

#include <string>

#include <ATen/ATen.h>
#include <ATen/native/autotune/api.h>
#include <ATen/native/autotune/dispatch/common.h>
#include <ATen/native/autotune/utils/common.h>
#include <c10/util/ArrayRef.h>

namespace autotune {
namespace kernels {

/*
Conv2D can be computed by one of three libraries:
  Native (TH)
  NNPack
  MKL

Native and MKL use Im2Col + GEMM [1], while NNPack selects between several
methods based on the input.

MKL additionally appears to have fused the Im2Col and GEMM portions to
pipeline compute and memory access; however it does incur additional layout
conversion cost, presumably due to the difference between native Tensor
memory layout and MKL format.

NNPack uses one of four algorithms:
  - For 1x1 convolutions, Im2Col is a no-op and NNPack directly calls GEMM.
  - For 3x3 convolutions, use Winograd. (2 FLOPS per element [2])
  - For kernels with either height or width greater than one up to 16x16,
    use FFT convolution. (4 FLOPS per element [2])
  - Otherwise, use indirect convolution to skip Im2Col. [3]


=============================================================================
== Im2Col + GEMM ============================================================
=============================================================================
This approach performs an (NxM) x (MxQ) matrix multiplication, where:
  N = image_h * image_w
  M = channels_in * kernel_h * kernel_w
  Q = channels_out

Because of the regular nature of matrix multiplication, modern CPUs can
vectorize along the inner dimension. If M < SIMD size then this simply
results in unutilized slots when the multiply-add occurs. Furthermore, modern
BLAS libraries scale better than the naive O(N * M * Q) matrix multiplication.
For instance, if define:
  b = min(N, M, Q)
and partition the matricies into blocks:
  (N/b * b x M/b * b) x (M/b * b x Q/b * b)
(assuming b divides N, M, and Q for simplicity) and use Strassen
multiplication on the blocks, we can achieve a `b ** 0.2` speedup. This
speedup turns out to agree quite well with the data, particularly for MKL.

Lastly, MKL seems to always try to vectorize the output channel dimension
such that the run time is insensitive so long as
  c_out < system::cpu_vector_size.

[1] https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
[2] https://www.reddit.com/r/MachineLearning/comments/4bswi6/nnpack_acceleration_package_for_neural_networks/d1cdv1y/
[3] https://arxiv.org/pdf/1907.02129v1.pdf
*/

// ============================================================================
// == Emperical parameters ====================================================
// ============================================================================
struct Conv2DPerfParams {
  api::Implementation impl;

  // Coefficient for overall efficiency which isn't represented elsewhere.
  // Lower is better.
  double overall_scale;

  // In clock cycles, and will be divided by system::cpu_hz.
  int64_t overhead;

  // effective_num_threads = num_threads ** thread_scaling.
  double thread_scaling;

  // Factors for data movement. (cycles per byte of input / intermediate state)
  double im2col_factor;
  double layout_factor;

  int64_t min_effective_c_out;
  bool supports_3x3_winograd;
  bool supports_fft;
};

/*  Non-trivial emperical parameters:
      conv2d_*_perf.overall_scale
      conv2d_*_perf.overhead

      conv2d_native_perf.thread_scaling
      conv2d_native_perf.im2col_factor
      conv2d_nnpack_perf.layout_factor
      conv2d_mkl_perf.thread_scaling
      conv2d_mkl_perf.layout_factor
      conv2d_mkl_perf.min_effective_c_out

    Total = 2 * 3 + 6
          = 12
*/

constexpr Conv2DPerfParams conv2d_native_perf{
    api::Implementation::kConv2D_Native,
    /*overall_scale=         */ 4,
    /*overhead=              */ 250'000,
    /*thread_scaling=        */ 0.5,
    /*im2col_factor=         */ 0.25,
    /*layout_factor=         */ 0,
    /*min_effective_c_out=   */ 1,
    /*supports_3x3_winograd= */ false,
    /*supports_fft=          */ false
};

constexpr int64_t nnpack_spectal_tile_threshold = 4;
constexpr Conv2DPerfParams conv2d_nnpack_perf{
    api::Implementation::kConv2D_NNPack,
    /*overall_scale=         */ 4,
    /*overhead=              */ 50'000,
    /*thread_scaling=        */ 0,
    /*im2col_factor=         */ 0,
    /*layout_factor=         */ 0.5,
    /*min_effective_c_out=   */ 1,
    /*supports_3x3_winograd= */ true,
    /*supports_fft=          */ true
};

constexpr Conv2DPerfParams conv2d_mkl_perf{
    api::Implementation::kConv2D_MKL,
    /*overall_scale=         */ 2.5,
    /*overhead=              */ 1'000'000,
    /*thread_scaling=        */ 0.8,
    /*im2col_factor=         */ 0,
    /*layout_factor=         */ 0.25,
    /*min_effective_c_out=   */ system::cpu_vector_size,
    /*supports_3x3_winograd= */ false,
    /*supports_fft=          */ false
};

struct ConvolutionArgs {
  at::Tensor input;
  at::Tensor weight;
  c10::IntArrayRef output_sizes;
  c10::IntArrayRef dilation;
  bool is_transposed;
  bool is_depthwise;
  int64_t groups;
  int num_threads;
};

class ConvolutionEntryPoint : public selection::KernelEntryPoint {
 public:
  using Args = ConvolutionArgs;
  ConvolutionEntryPoint(const ConvolutionArgs&);
  bool fallback() override;
  api::Task task() override;
  cost_estimates costs() override;
  supported_implementations implementations() override;
  std::string repr() override;

 private:
  c10::IntArrayRef input_sizes_;
  c10::IntArrayRef weight_sizes_;
  c10::IntArrayRef output_sizes_;
  size_t itemsize_;
  bool fallback_;
  int num_threads_;
  bool skip_nnpack_ = false;
  double cost(const Conv2DPerfParams&);
};
} // namespace kernels
} // namespace autotune
