#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>

#include <ATen/native/autotune/definitions.h>

namespace cost {

// Placeholder. These should be determined on a system by system basis.
namespace main_memory {
namespace bandwidth {
static int64_t sequential_read = 7'000'000'000;
static int64_t random_read = 1'600'000'000;
static int64_t sequential_write = 5'000'000'000;
static int64_t random_write = 400'000'000;
} // namespace bandwidth

static double approx_latency = 10.0 * 1e-9;
} // namespace main_memory
static double cpu_hz = 2'394'444'000;
static int64_t cpu_vector_size = 16; // Broadwell, fp32
static int64_t cache_line_size = 64;

autotune::EntryPoint dispatch_conv(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::IntArrayRef output_sizes,
    c10::IntArrayRef dilation,
    bool is_transposed,
    bool is_depthwise,
    int64_t groups);

} // namespace cost
