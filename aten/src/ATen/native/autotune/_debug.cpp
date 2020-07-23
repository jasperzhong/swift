#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <thread>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
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
  int64_t c_out = 8;
  at::globalContext().setUserEnabledCost(true);

  // TODO: Investigate grevious stalls in multithreaded runs.
  at::set_num_threads(1);

  for (int64_t k = 0; k < 5; k++){
    std::cout << "======================\n" << std::endl;

    int extra_for_warmup = !k ? 100 : 0;
    for (int i = 0; i < 100 + extra_for_warmup; i++){
      int64_t input_h = k * 2 + s;
      int64_t input_w = k * 2 + s;

      at::Tensor x = at::ones({1, c, input_h, input_w});
      at::Tensor w = at::ones({c_out, c, 3, 3});
      at::Tensor b = at::ones({c_out});

      std::vector<int64_t> output_sizes {1, c_out, input_h - 2, input_w - 2};
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



      size_t delta_ns_0;
      size_t delta_ns_1;
      std::chrono::time_point<std::chrono::high_resolution_clock> t1;
      {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto choice = autotune::CostDispatcher::singleton().choose(e);
        delta_ns_0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - t0).count();

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
        t1 = std::chrono::high_resolution_clock::now();
        choice.finished();
        delta_ns_1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - t1).count();
      }


      // Note to self: debug print statements are expensive.
      // std::cout << "Overhead: " << delta_ns_0 << "   " << delta_ns_1 << std::endl;


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
