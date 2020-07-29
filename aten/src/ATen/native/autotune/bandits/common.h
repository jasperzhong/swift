#pragma once

#include <memory>
#include <random>

#include <ATen/native/autotune/dispatch/common.h>
#include <ATen/native/autotune/kernels/common.h>

#include <c10/util/Exception.h>

namespace autotune {
namespace bandits {
class Bandit {
 public:
  virtual kernels::Implementation choose() = 0;
  virtual void update(kernels::Implementation choice, size_t delta_ns) = 0;

  kernels::Implementation choose_safe(
      const selection::KernelEntryPoint::supported_implementations
          implementations);
  selection::KernelEntryPoint::supported_implementations implementations();

  // This struct is meant to contain any running information about an
  // implementation that will be shared by all bandits of a given type. (e.g.
  // statistics). Bandit subclasses should override this struct.
  struct ImplState final {
    virtual void f() = 0; // Force pure virtual struct;
  };

 protected:
  Bandit(selection::KernelEntryPoint::cost_estimates& costs, unsigned seed);
  std::mt19937 engine_;

 private:
  Bandit(unsigned seed) : engine_(seed){};
  selection::KernelEntryPoint::supported_implementations implementations_;
};
} // namespace bandits
} // namespace autotune
