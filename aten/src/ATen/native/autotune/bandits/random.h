#pragma once

#include <ATen/native/autotune/api.h>
#include <ATen/native/autotune/bandits/common.h>
#include <ATen/native/autotune/dispatch/common.h>

namespace autotune {
namespace bandits {
class DrunkenBandit : public Bandit {
 public:
  DrunkenBandit(
      selection::KernelEntryPoint::cost_estimates& costs,
      unsigned seed);
  api::Implementation choose() override;
  void update(api::Implementation, size_t) override;
};
} // namespace bandits
} // namespace autotune
