#pragma once

#include <c10/util/ArrayRef.h>
#include <ATen/ATen.h>
#include <ATen/native/autotune/bandit.h>


namespace autotune {

enum class Conv2D_Dispatch { Native, MKLDNN, Fallback, Unsupported };

std::pair<Conv2D_Dispatch, bandits::Bandit::Callback> dispatch_conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::IntArrayRef output_size,
    c10::IntArrayRef dilation,
    bool is_transposed,
    bool is_depthwise,
    int64_t groups);

}  // autotune
