#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <utility>

#include <c10/util/ArrayRef.h>
#include <ATen/ATen.h>

#include <ATen/native/autotune/bandit.h>
#include <ATen/native/autotune/definitions.h>
#include <ATen/native/autotune/estimator.h>


namespace autotune {

class CostDispatcher {
 public:
  static CostDispatcher& singleton(){
    static CostDispatcher _singleton;
    return _singleton;
  }

  class RAII_Choice {
   public:
    RAII_Choice(const RAII_Choice&) = delete;
    RAII_Choice(RAII_Choice&&) = default;
    RAII_Choice(cache_key key, DispatchChoice choice, bandits::GaussianBandit* b);
    ~RAII_Choice();
    DispatchChoice get() { return choice_; }

   private:
    cache_key key_;
    DispatchChoice choice_;
    bandits::GaussianBandit* b_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  };
  RAII_Choice choose(entry_point e);

 private:
  CostDispatcher() = default;
  std::map<cache_key, bandits::GaussianBandit> bandits_;
  uint64_t next_seed_ = 0;
};


entry_point dispatch_conv(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::IntArrayRef output_sizes,
    c10::IntArrayRef dilation,
    bool is_transposed,
    bool is_depthwise,
    int64_t groups);

















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
