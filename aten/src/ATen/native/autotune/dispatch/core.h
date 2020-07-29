#pragma once

#include <chrono>
#include <functional>
#include <type_traits>

#include <ATen/native/autotune/bandits/common.h>
#include <ATen/native/autotune/dispatch/common.h>
#include <ATen/native/autotune/kernels/common.h>
#include <ATen/native/autotune/kernels/convolution.h>

namespace autotune {
namespace selection {

// Holds active Bandits, as well as any running statistics that may be
// needed for the bandits or debugging.
template <typename T>
class DispatchWorkingState {
 public:
  static_assert(std::is_base_of<bandits::Bandit, T>::value);
  static DispatchWorkingState<T>& singleton() {
    static DispatchWorkingState<T> _singleton;
    return _singleton;
  }

  kernels::Implementation choose(
      KernelEntryPoint::map_key,
      KernelEntryPoint::supported_implementations,
      std::function<KernelEntryPoint::cost_estimates()>);

  void update(KernelEntryPoint::map_key, kernels::Implementation, size_t);

  std::shared_ptr<typename T::ImplState> bandit_state(
      kernels::Implementation impl) {
    return bandit_state_[static_cast<size_t>(impl)];
  }

 private:
  DispatchWorkingState();
  friend class DispatchInterface;
  std::map<KernelEntryPoint::map_key, std::unique_ptr<T>> bandits_;
  std::
      array<std::shared_ptr<typename T::ImplState>, kernels::NumImplementations>
          bandit_state_;
  size_t next_seed_{0};
};

class DispatchInterface {
 public:
  static DispatchInterface& singleton() {
    static DispatchInterface _singleton;
    return _singleton;
  }

  enum class AvailableBandits {
    kRandomChoice,
    kGaussian,

    kNone,
  };

  AvailableBandits active_bandit();
  void setActiveBandit(AvailableBandits);

  kernels::Implementation choose(
      DispatchInterface::AvailableBandits,
      KernelEntryPoint::map_key,
      KernelEntryPoint::supported_implementations,
      std::function<KernelEntryPoint::cost_estimates()>);
  void update(
      DispatchInterface::AvailableBandits,
      KernelEntryPoint::map_key,
      kernels::Implementation,
      size_t);

 private:
  AvailableBandits active_bandit_{AvailableBandits::kRandomChoice};
};

template <typename T>
class SelectImplementation {
 public:
  static_assert(std::is_base_of<KernelEntryPoint, T>::value);

  // In theory, it's possible to forward args to T's constructor using
  // template <typename... Args> and std::forward. In practice getting
  // that to compile and link is a nightmare, so instead we simply have
  // each task define an Args struct.
  SelectImplementation(typename T::Args);

  kernels::Implementation choice();
  void finish();

 private:
  T entry_point_;
  DispatchInterface::AvailableBandits bandit_type_{
      DispatchInterface::AvailableBandits::kNone};
  kernels::Implementation choice_;
  bool record_duration_{false};
  bool record_finished_{false};
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

template class SelectImplementation<kernels::ConvolutionEntryPoint>;
using DispatchConvolution =
    SelectImplementation<kernels::ConvolutionEntryPoint>;

} // namespace selection
} // namespace autotune
