#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/native/autotune/dispatch/common.h>
#include <c10/util/ArrayRef.h>

namespace autotune {
namespace kernels {

static double convolution_overhead = 20.0e-6; // 20 us.

struct ConvolutionArgs {
  at::Tensor input;
  at::Tensor weight;
  c10::IntArrayRef output_sizes;
  c10::IntArrayRef dilation;
  bool is_transposed;
  bool is_depthwise;
  int64_t groups;
};

class ConvolutionEntryPoint : public selection::KernelEntryPoint {
 public:
  using Args = ConvolutionArgs;
  ConvolutionEntryPoint(const ConvolutionArgs&);
  bool fallback() override;
  Task task() override;
  cost_estimates costs() override;
  supported_implementations implementations() override;
  std::string repr() override;

 private:
  c10::IntArrayRef input_sizes_;
  c10::IntArrayRef input_strides_;
  c10::IntArrayRef weight_sizes_;
  c10::IntArrayRef weight_strides_;
  c10::IntArrayRef output_sizes_;
  size_t itemsize_;
  bool fallback_;
};
} // namespace kernels
} // namespace autotune
