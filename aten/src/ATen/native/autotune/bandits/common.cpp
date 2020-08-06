#include <ATen/native/autotune/bandits/common.h>

#include <ATen/native/autotune/api.h>
#include <ATen/native/autotune/dispatch/common.h>

namespace autotune {
namespace bandits {

Bandit::Bandit(
    selection::KernelEntryPoint::cost_estimates& costs,
    unsigned seed)
    : Bandit(seed) {
  // Used to check dispatch correctness.
  for (auto c : costs) {
    implementations_.push_back(c.impl);
  }
};

void Bandit::summarize(selection::KernelEntryPoint::MapKey) {}

const selection::KernelEntryPoint::supported_implementations& Bandit::
    implementations() const {
  return implementations_;
}
} // namespace bandits
} // namespace autotune
