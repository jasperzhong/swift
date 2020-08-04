#pragma once

#include <memory>
#include <random>
#include <string>

#include <ATen/native/autotune/api.h>
#include <ATen/native/autotune/dispatch/common.h>

#include <c10/util/Exception.h>

namespace autotune {
namespace bandits {

class Bandit {
 public:
  virtual api::Implementation choose() = 0;
  virtual void update(api::Implementation choice, size_t delta_ns) = 0;
  virtual void summarize(selection::KernelEntryPoint::MapKey);

  api::Implementation choose_safe(
      const selection::KernelEntryPoint::supported_implementations
          implementations);
  selection::KernelEntryPoint::supported_implementations implementations();

 protected:
  Bandit(selection::KernelEntryPoint::cost_estimates& costs, unsigned seed);
  std::mt19937 engine_;

 private:
  Bandit(unsigned seed) : engine_(seed){};
  selection::KernelEntryPoint::supported_implementations implementations_;
};
} // namespace bandits
} // namespace autotune
