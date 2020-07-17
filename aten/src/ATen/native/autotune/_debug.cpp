#include <chrono>
#include <iostream>
#include <thread>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/native/autotune/dispatch.h>

namespace at {
namespace native {
bool autotune_debug() {
  at::globalContext().setUserEnabledCost(true);
  at::Tensor x = at::ones({1, 3, 256, 256});
  at::Tensor w = at::ones({1, 1, 3, 3});

  auto e = autotune::dispatch_conv(
    /*input=*/x,
    /*weight=*/w,
    /*output_sizes=*/{1, 1, 254, 254},
    /*dilation=*/{1, 1},
    /*is_transposed=*/false,
    /*is_depthwise=*/false,
    /*groups=*/1
  );


  for (int i = 0; i < 30; i++){
    {
      auto choice = autotune::CostDispatcher::singleton().choose(e);
      switch (choice.get()) {
        case autotune::DispatchChoice::kConv2D_Native:
          std::this_thread::sleep_for (std::chrono::microseconds(1000));
          break;
        case autotune::DispatchChoice::kConv2D_MKL:
          std::this_thread::sleep_for (std::chrono::microseconds(900));
          break;
        default:
          TORCH_INTERNAL_ASSERT(false, "invalid autotune choice.")
      }
    }
  }


  return false;
}
} // namespace native
} // namespace at
