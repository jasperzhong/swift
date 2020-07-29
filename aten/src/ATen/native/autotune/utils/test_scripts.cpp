#include <ATen/native/autotune/utils/test_scripts.h>

#include <random>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/autotune/api.h>

using DispatchInterface = autotune::selection::DispatchInterface;
using Implementation = autotune::kernels::Implementation;
using DispatchConvolution = autotune::selection::DispatchConvolution;

namespace autotune {
at::Tensor convolution(at::Tensor x, at::Tensor weight) {
  at::Tensor output;
  std::vector<int64_t> stride{1, 1};
  std::vector<int64_t> dilation{1, 1};
  std::vector<int64_t> padding{0, 0};
  std::vector<int64_t> output_padding{0, 0};
  auto output_sizes = at::native::conv_output_size(
      x.sizes(), weight.sizes(), padding, stride, dilation);
  auto bias = at::ones({output_sizes[1]});

  {
    auto dispatch = DispatchConvolution( //
        {/*input=         */ x,
         /*weight=        */ weight,
         /*output_sizes=  */ output_sizes,
         /*dilation=      */ dilation,
         /*is_transposed= */ false,
         /*is_depthwise=  */ false,
         /*groups=        */ 1});

    switch (dispatch.choice()) {
      case Implementation::kConv2D_Native:
        output = at::_convolution_nogroup(
            x.contiguous(),
            weight,
            bias,
            stride,
            padding,
            dilation,
            /*transposed=*/false,
            output_padding);
        break;
      case Implementation::kConv2D_NNPack:
        output = at::_nnpack_spatial_convolution(
            x, weight, bias, padding, stride);
        break;
      case Implementation::kConv2D_MKL:
        output = at::mkldnn_convolution(
            x.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            padding,
            stride,
            dilation,
            /*groups=*/1);
        break;
      default:
        TORCH_INTERNAL_ASSERT(false);
        break;
    }
    dispatch.finish();
  }

  return x;
}

void set_no_bandit(){
  at::globalContext().setUserEnabledAutotune(false);
  DispatchInterface::singleton().setActiveBandit(
      DispatchInterface::AvailableBandits::kNone);
}
void set_drunken_bandit(){
  at::globalContext().setUserEnabledAutotune(true);
  DispatchInterface::singleton().setActiveBandit(
      DispatchInterface::AvailableBandits::kRandomChoice);
}
void set_gaussian_bandit(){
  at::globalContext().setUserEnabledAutotune(true);
  DispatchInterface::singleton().setActiveBandit(
      DispatchInterface::AvailableBandits::kGaussian);
}

void flush_results() {
  logging::flush();
}

void test_bandit(int n) {
  // This will also initialize NNPack.
  TORCH_INTERNAL_ASSERT(at::_nnpack_available());

  // c, c_out, input_h, input_w, kernel_h (and w)
  std::vector<std::vector<int64_t>> sizes = {
      {16, 8, 128, 128, 1},
      {32, 16, 64, 64, 1},
      {32, 64, 32, 32, 1},
      {128, 16, 32, 32, 1},
      {64, 32, 32, 32, 1},
      {128, 64, 16, 16, 1},
      {256, 128, 8, 8, 1},
      {512, 256, 4, 4, 1},

      {16, 8, 128, 128, 3},
      {32, 16, 64, 64, 3},
      {32, 64, 32, 32, 3},
      {128, 16, 32, 32, 3},
      {64, 32, 32, 32, 3},
      {128, 64, 16, 16, 3},
      {256, 128, 8, 8, 3},
      {512, 256, 4, 4, 3}
  };

  std::mt19937 engine(0);
  std::uniform_int_distribution<size_t> distribution(0, sizes.size() - 1);
  for (size_t i = 0; i < n; i++) {
    auto choice = i < sizes.size() ? i : distribution(engine);
    int64_t c = sizes[choice][0];
    int64_t c_out = sizes[choice][1];
    int64_t input_h = sizes[choice][2];
    int64_t input_w = sizes[choice][3];
    int64_t kernel_size = sizes[choice][4];
    convolution(
        at::ones({1, c, input_h, input_w}),
        at::ones({c_out, c, kernel_size, kernel_size}));
  }
}
} // namespace autotune
