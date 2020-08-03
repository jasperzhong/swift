#pragma once

#include <array>
#include <chrono>
#include <functional>
#include <type_traits>

#include <ATen/native/autotune/api.h>
#include <ATen/native/autotune/bandits/common.h>
#include <ATen/native/autotune/dispatch/common.h>
#include <ATen/native/autotune/kernels/convolution.h>

namespace autotune {
namespace selection {

class DispatchInterface {
 public:
  static DispatchInterface& singleton() {
    static DispatchInterface _singleton;
    return _singleton;
  }

  api::AvailableBandits active_bandit();
  void setActiveBandit(api::AvailableBandits);

  api::Implementation choose(
      api::AvailableBandits,
      KernelEntryPoint::MapKey,
      KernelEntryPoint::supported_implementations,
      std::function<KernelEntryPoint::cost_estimates()>);

  void update(
      api::AvailableBandits,
      KernelEntryPoint::MapKey,
      api::Implementation,
      size_t);

  size_t times_chosen(api::Implementation);

 private:
  DispatchInterface() {};
  api::AvailableBandits active_bandit_{api::AvailableBandits::kNone};
  std::array<size_t, api::NumImplementations> chosen_counts_;
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

  api::Implementation choice();
  void finish();

 private:
  T entry_point_;
  api::AvailableBandits bandit_type_;
  api::Implementation choice_;
  bool record_duration_{false};
  bool record_finished_{false};
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

template class SelectImplementation<kernels::ConvolutionEntryPoint>;
using DispatchConvolution =
    SelectImplementation<kernels::ConvolutionEntryPoint>;

} // namespace selection
} // namespace autotune
