#include <ATen/native/autotune/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <c10/util/ArrayRef.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/native/autotune/bandit.h>
#include <ATen/native/autotune/cost.h>


namespace autotune {

static entry_point FALLBACK {{0, 0}, [](){ return cost_priors({{DispatchChoice::kFallback, 0.0}}); }};
static entry_point UNSUPPORTED {{0, 0}, [](){ return cost_priors({{DispatchChoice::kUnsupported, 0.0}}); }};

CostDispatcher::RAII_Choice::RAII_Choice(cache_key key, DispatchChoice choice, bandits::GaussianBandit* b)
  : key_(key), choice_(choice), b_(b) {
  start_ = std::chrono::high_resolution_clock::now();
}

CostDispatcher::RAII_Choice::~RAII_Choice(){
  auto end = std::chrono::high_resolution_clock::now();
  auto delta_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();

  //TODO: remove `key_` once I remove this print line.
  std::cout << "~RAII_Choice: " << key_ << "  " << static_cast<size_t>(choice_) << "  " << delta_ns << std::endl;

  b_->update(choice_, (double)delta_ns / 1.0e9);
}

CostDispatcher::RAII_Choice CostDispatcher::choose(entry_point e){
  auto key = e.first;
  auto prior_fn = e.second;

  if (bandits_.find(key) == bandits_.end()){
    bandits_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(key),
      std::forward_as_tuple(prior_fn(), next_seed_));
    next_seed_++;
  }

  auto& bandit = bandits_.at(key);
  auto choice = bandit.select();

  return {key, choice, &bandit};
}

// https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
void hash_combine(cache_key & seed, c10::IntArrayRef v){
    for (auto vi : v) {
      seed.first ^= std::hash<int64_t>{}(vi) + 0x9e3779b9 + (seed.first << 6) + (seed.first >> 2);
      seed.second ^= std::hash<int64_t>{}(vi) + 0x9e3779b9 + (seed.second << 6) + (seed.second >> 2);
    }
}

entry_point dispatch_conv(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::IntArrayRef output_sizes,
    c10::IntArrayRef dilation,
    bool is_transposed,
    bool is_depthwise,
    int64_t groups) {

  if (!at::globalContext().userEnabledCost())
    return FALLBACK;

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
    return UNSUPPORTED;

  c10::IntArrayRef input_sizes(input.sizes());
  c10::IntArrayRef input_strides(input.strides());
  c10::IntArrayRef weight_sizes(weight.sizes());
  c10::IntArrayRef weight_strides(weight.strides());

  cache_key key {0, 1};
  std::vector<c10::IntArrayRef> features{
      input_sizes, input_strides, weight_sizes, weight_strides, output_sizes};
  for (auto i : features)
    hash_combine(key, i);

  auto cost_prior_fn = [=]() {
    return cost::conv_priors(
        input_sizes,
        input_strides,
        weight_sizes,
        weight_strides,
        output_sizes,
        input.itemsize());
  };

  return {key, cost_prior_fn};
}
















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
