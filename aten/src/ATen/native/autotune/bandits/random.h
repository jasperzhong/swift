#pragma once

#include <ATen/native/autotune/bandits/common.h>
#include <ATen/native/autotune/dispatch/common.h>

namespace autotune {
namespace bandits {
class DrunkenBandit : public Bandit {
 public:
  DrunkenBandit(
      selection::KernelEntryPoint::cost_estimates& costs,
      unsigned seed);
  kernels::Implementation choose() override;
  void update(kernels::Implementation, size_t) override;

  struct ImplState {};

 private:
  selection::KernelEntryPoint::supported_implementations implementations_;
};
} // namespace bandits
} // namespace autotune
