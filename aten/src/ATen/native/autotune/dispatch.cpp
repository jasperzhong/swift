#include <ATen/native/autotune/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <c10/util/ArrayRef.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/native/autotune/bandit.h>
#include <ATen/native/autotune/cost.h>


namespace autotune {

class AutoTuner {
 public:
  enum class Specialization { kConv2D };

  static AutoTuner& conv2d_singleton() {
    static AutoTuner _conv2d_singleton(Specialization::kConv2D, /*impl_count=*/2);
    return _conv2d_singleton;
  }

  bandits::Bandit::Callback get(size_t key, std::function<std::vector<double>()> priors, std::string repr);

 private:
  AutoTuner(Specialization specialization, int64_t impl_count): specialization_(specialization) {}
  using cache_key = std::pair<Specialization, size_t>;
  Specialization specialization_;
  uint64_t next_seed_ = 0;

  std::map<cache_key, std::unique_ptr<bandits::Bandit>> cache_;
};

bandits::Bandit::Callback AutoTuner::get(size_t key, std::function<std::vector<double>()> priors, std::string repr) {
    // TODO: Don't compute repr on every pass.
    cache_key key_pair {specialization_, key};
    if (cache_.find(key_pair) == cache_.end()) {
        cache_.emplace(key_pair, std::make_unique<bandits::Bandit>(priors(), next_seed_, repr));
        next_seed_++;
    }
    return cache_.find(key_pair)->second->sample();
}

std::pair<Conv2D_Dispatch, bandits::Bandit::Callback> dispatch_conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::IntArrayRef output_size,
    c10::IntArrayRef dilation,
    bool is_transposed,
    bool is_depthwise,
    int64_t groups) {
  auto start_time = std::chrono::high_resolution_clock::now();
  if (!at::globalContext().userEnabledCost())
    return {Conv2D_Dispatch::Fallback, bandits::Bandit::Callback::make_empty()};

  // Autotuning is only prototyped on a subset of Conv2D.
  bool supported = (
    input.options().backend() == at::Backend::CPU &&
    input.scalar_type() == at::kFloat &&
    weight.scalar_type() == at::kFloat &&
    weight.ndimension() == 4 &&
    !is_transposed &&
    !is_depthwise &&
    groups == 1 &&
    std::all_of(dilation.begin(), dilation.end(), [](int64_t x){ return x == 1; })
  );
  if (!supported)
    return {Conv2D_Dispatch::Unsupported, bandits::Bandit::Callback::make_empty()};

  auto roofline = cost::conv2d::Roofline(input, weight, output_size);
  auto callback = AutoTuner::conv2d_singleton().get(roofline.key(), [&roofline](){ return roofline.compute(); }, roofline.repr());

  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto delta_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

  return {callback.choice_ == 0 ? Conv2D_Dispatch::Native : Conv2D_Dispatch::MKLDNN, callback};
}

}  // autotune
