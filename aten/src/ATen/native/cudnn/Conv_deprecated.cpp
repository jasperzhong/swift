#include <ATen/ATen.h>
#include <ATen/native/ConvUtils.h>

namespace at { namespace native {

// TODO (@zasdfgbnm): this is here only for compatibility, remove this in the future
Tensor cudnn_convolution_deprecated(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias /* optional */,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  auto output = at::cudnn_convolution(input, weight, padding, stride, dilation, groups, benchmark, deterministic);
  if (bias.defined()) {
    output = output + reshape_bias(input.dim(), bias);
  }
  return output;
}

// TODO (@zasdfgbnm): this is here only for compatibility, remove this in the future
Tensor cudnn_convolution_deprecated2(
    const Tensor& input_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  return at::cudnn_convolution(input_t, weight_t, padding, stride, dilation, groups, benchmark, deterministic, at::globalContext().allowTF32CuDNN());
}

// TODO (@zasdfgbnm): this is here only for compatibility, remove this in the future
Tensor cudnn_convolution_transpose_deprecated(
    const Tensor& input, const Tensor& weight, const Tensor& bias /* optional */,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  auto output = at::cudnn_convolution_transpose(input, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  if (bias.defined()) {
    output = output + reshape_bias(input.dim(), bias);
  }
  return output;
}

// TODO (@zasdfgbnm): this is here only for compatibility, remove this in the future
Tensor cudnn_convolution_transpose_deprecated2(
    const Tensor& input_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
    return at::cudnn_convolution_transpose(input_t, weight_t, padding, output_padding, stride, dilation, groups, benchmark, deterministic, at::globalContext().allowTF32CuDNN());
}

}}  // namespace at::native