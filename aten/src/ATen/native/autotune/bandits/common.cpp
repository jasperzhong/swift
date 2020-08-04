#include <ATen/native/autotune/bandits/common.h>

#include <ATen/native/autotune/api.h>

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

api::Implementation Bandit::choose_safe(
    const selection::KernelEntryPoint::supported_implementations
        implementations) {
  TORCH_INTERNAL_ASSERT(implementations.size() == implementations_.size());
  for (size_t i = 0; i < implementations.size(); i++) {
    TORCH_INTERNAL_ASSERT(implementations[i] == implementations_[i]);
  }
  return choose();
}

selection::KernelEntryPoint::supported_implementations Bandit::
    implementations() {
  return implementations_;
}

} // namespace bandits
} // namespace autotune
