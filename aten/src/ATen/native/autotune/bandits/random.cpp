#include <ATen/native/autotune/bandits/random.h>

#include <random>

#include <ATen/native/autotune/api.h>
#include <ATen/native/autotune/dispatch/common.h>

namespace autotune {
namespace bandits {
DrunkenBandit::DrunkenBandit(
    selection::KernelEntryPoint::cost_estimates& costs,
    unsigned seed)
    : Bandit(costs, seed) {}

api::Implementation DrunkenBandit::choose() {
  auto n = implementations().size();
  std::uniform_int_distribution<size_t> distribution(0, n - 1);
  auto choice = distribution(engine_);
  return implementations()[choice];
}

void DrunkenBandit::update(api::Implementation, size_t) {}

} // namespace bandits
} // namespace autotune
