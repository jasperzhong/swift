#include <ATen/native/autotune/bandits/random.h>

#include <random>

#include <ATen/native/autotune/dispatch/common.h>

namespace autotune {
namespace bandits {
DrunkenBandit::DrunkenBandit(
    selection::KernelEntryPoint::cost_estimates& costs,
    unsigned seed)
    : Bandit(costs, seed) {}

kernels::Implementation DrunkenBandit::choose() {
  std::uniform_int_distribution<size_t> distribution(
      0, implementations().size() - 1);
  auto choice = distribution(engine_);
  return implementations()[choice];
}

void DrunkenBandit::update(kernels::Implementation, size_t) {}

} // namespace bandits
} // namespace autotune
