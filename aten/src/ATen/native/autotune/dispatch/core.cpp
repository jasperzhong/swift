#include <ATen/native/autotune/dispatch/core.h>

#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <ATen/Context.h>
#include <ATen/native/autotune/bandits/common.h>
#include <ATen/native/autotune/bandits/gaussian.h>
#include <ATen/native/autotune/bandits/random.h>
#include <ATen/native/autotune/dispatch/logging.h>
#include <ATen/native/autotune/kernels/common.h>
#include <c10/util/Exception.h>

namespace autotune {
namespace selection {

template <typename T>
DispatchWorkingState<T>::DispatchWorkingState() {
  for (size_t i = 0; i < bandit_state_.size();i++){
    bandit_state_[i] = std::make_shared<typename T::ImplState>();
  }
}

template <typename T>
kernels::Implementation DispatchWorkingState<T>::choose(
    KernelEntryPoint::map_key key,
    KernelEntryPoint::supported_implementations supported_implementations,
    std::function<KernelEntryPoint::cost_estimates()> cost_fn) {
  if (bandits_.find(key) == bandits_.end()) {
    auto cost_estimates = cost_fn();
    bandits_[key] = std::make_unique<T>(cost_estimates, next_seed_);
    next_seed_++;
  }

  return bandits_.at(key)->choose_safe(
      // The bandit will check that its supported_implementations matches.
      supported_implementations);
}

template <typename T>
void DispatchWorkingState<T>::update(
    KernelEntryPoint::map_key key,
    kernels::Implementation impl,
    size_t delta_ns) {
  bandits_.at(key)->update(impl, delta_ns);
}

DispatchInterface::AvailableBandits DispatchInterface::active_bandit() {
  return active_bandit_;
}

void DispatchInterface::setActiveBandit(AvailableBandits b) {
  active_bandit_ = b;
}

kernels::Implementation DispatchInterface::choose(
    DispatchInterface::AvailableBandits bandit,
    KernelEntryPoint::map_key key,
    KernelEntryPoint::supported_implementations implementations,
    std::function<KernelEntryPoint::cost_estimates()> cost_estimates) {
  switch (bandit) {
    case AvailableBandits::kRandomChoice:
      return DispatchWorkingState<bandits::DrunkenBandit>::singleton().choose(
          key, implementations, cost_estimates);
    case AvailableBandits::kGaussian:
      return DispatchWorkingState<bandits::GaussianBandit>::singleton().choose(
          key, implementations, cost_estimates);
    default:
      TORCH_INTERNAL_ASSERT(false, "Could not select bandit.")
  }
}

void DispatchInterface::update(
      DispatchInterface::AvailableBandits bandit,
      KernelEntryPoint::map_key key,
      kernels::Implementation choice,
      size_t delta_ns) {
  switch (bandit) {
    case AvailableBandits::kRandomChoice:
      return DispatchWorkingState<bandits::DrunkenBandit>::singleton().update(
          key, choice, delta_ns);
    case AvailableBandits::kGaussian:
      return DispatchWorkingState<bandits::GaussianBandit>::singleton().update(
          key, choice, delta_ns);
    default:
      TORCH_INTERNAL_ASSERT(false, "Could not select bandit.")
  }
}

template <typename T>
SelectImplementation<T>::SelectImplementation(typename T::Args args)
    : entry_point_(args) {
  if (!at::globalContext().userEnabledAutotune()) {
    choice_ = kernels::Implementation::kDisabled;
  } else if (entry_point_.fallback()) {
    choice_ = kernels::Implementation::kFallback;
  } else {
    auto available_implementations = entry_point_.implementations();
    TORCH_INTERNAL_ASSERT(
        available_implementations.size(),
        "Autotuning is enabled and kernel did not request a fallback, "
        "however no implemenations are available.");

    bandit_type_ = DispatchInterface::singleton().active_bandit();
    choice_ = DispatchInterface::singleton().choose(
        bandit_type_, entry_point_.key(), available_implementations, [&]() {
          return entry_point_.costs();
        });

    record_duration_ = true;
    start_ = std::chrono::high_resolution_clock::now();
  }
}

template <typename T>
kernels::Implementation SelectImplementation<T>::choice() {
  return choice_;
}

template <typename T>
void SelectImplementation<T>::finish() {
  TORCH_INTERNAL_ASSERT(record_duration_ && !record_finished_);
  auto delta_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::high_resolution_clock::now() - start_)
                      .count();
  auto key = entry_point_.key();
  DispatchInterface::singleton().update(bandit_type_, key, choice_, delta_ns);
  logging::register_key(key, [&]() { return entry_point_.repr(); });
  logging::record(bandit_type_, key, choice_, delta_ns);
}

} // namespace selection
} // namespace autotune
