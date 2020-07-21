#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <thread>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/native/autotune/cost.h>
#include <ATen/native/autotune/dispatch.h>

namespace at {
namespace native {
bool autotune_debug() {

  std::mt19937 engine(0);
  std::uniform_int_distribution<int> distribution(0,1);

  int64_t s = 4;
  int64_t c = 64;
  at::globalContext().setUserEnabledCost(true);

  for (int64_t k = 0; k < 10; k++){
    std::cout << "======================\n" << std::endl;


    for (int i = 0; i < 10; i++){
      at::Tensor x = at::ones({1, c, k + s, k + s});
      at::Tensor w = at::ones({1, c, 3, 3});
      at::Tensor b = at::ones({1});

      std::vector<int64_t> output_sizes {1, 1, k + s - 2, k + s - 2};
      std::vector<int64_t> stride {1, 1};
      std::vector<int64_t> dilation {1, 1};
      std::vector<int64_t> padding {0, 0};
      std::vector<int64_t> output_padding {0, 0};
      auto e = cost::dispatch_conv(
        /*input=*/x,
        /*weight=*/w,
        /*output_sizes=*/output_sizes,
        /*dilation=*/{1, 1},
        /*is_transposed=*/false,
        /*is_depthwise=*/false,
        /*groups=*/1
      );

      size_t delta_ns;
      {
        auto choice = autotune::CostDispatcher::singleton().choose(e);
        auto t0 = std::chrono::high_resolution_clock::now();
        switch (choice.get()) {
          case autotune::DispatchChoice::kConv2D_Native:
            at::_convolution_nogroup(
              x.contiguous(), w, b,
              /*stride=*/{1, 1}, /*padding=*/{0, 0}, /*dilation=*/{1, 1},
              /*transposed=*/false, /*output_padding=*/{0, 0});
            break;
          case autotune::DispatchChoice::kConv2D_MKL:
            at::mkldnn_convolution(
              x.contiguous(), w.contiguous(), b.contiguous(),
              /*padding=*/{0, 0}, /*stride=*/{1, 1}, /*dilation=*/{1, 1}, /*groups=*/1);
            break;
          case autotune::DispatchChoice::kFallback:
            break;
          default:
            TORCH_INTERNAL_ASSERT(false, "invalid autotune choice.")
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        delta_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      }
      std::cout << delta_ns << std::endl;




      // auto choice = distribution(engine);
      // auto t0 = std::chrono::high_resolution_clock::now();
      // if (choice == 0) {
      //   at::thnn_conv2d(
      //         x, w, {3, 3}, b, stride, padding);
      // } else {
      //   at::mkldnn_convolution(
      //     x.contiguous(), w.contiguous(), b.contiguous(),
      //     padding, stride, dilation, /*groups=*/1);
      // }
      // auto t1 = std::chrono::high_resolution_clock::now();
      // auto delta_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      // printf("%d  %8.1f\n", choice, (double)delta_ns / 1.0e3);
    }

  }

  return false;
}
} // namespace native
} // namespace at
