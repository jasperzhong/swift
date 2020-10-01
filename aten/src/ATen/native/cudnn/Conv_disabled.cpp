#include <ATen/cuda/CUDAConfig.h>

#if !AT_CUDNN_ENABLED()

#include <ATen/ATen.h>

namespace at { namespace native {

// See Note [ATen preprocessor philosophy]

at::Tensor cudnn_convolution(
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("cudnn_convolution: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_backward_input(
    IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("cudnn_convolution_backward_input: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("cudnn_convolution_backward_weight: ATen not compiled with cuDNN support");
}

std::tuple<at::Tensor,at::Tensor> cudnn_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32, std::array<bool,2> output_mask) {
  AT_ERROR("cudnn_convolution_backward: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_transpose(
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("cudnn_convolution_transpose: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_transpose_backward_input(
    const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("cudnn_convolution_transpose_backward: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_transpose_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("cudnn_convolution_transpose_backward_weight: ATen not compiled with cuDNN support");
}

std::tuple<at::Tensor,at::Tensor> cudnn_convolution_transpose_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32, std::array<bool,2> output_mask) {
  AT_ERROR("cudnn_convolution_transpose_backward: ATen not compiled with cuDNN support");
}

}}

#endif  // AT_CUDNN_ENABLED