#pragma once

#include <map>

#include <ATen/native/autotune/api.h>
#include <ATen/native/autotune/bandits/common.h>
#include <ATen/native/autotune/dispatch/common.h>
#include <ATen/native/autotune/utils/stats.h>

namespace autotune {
namespace bandits {

static int64_t thompson_k(size_t count) {
  return (count + 1 > 4) ? 4 : count + 1;
}

static size_t warmup = 50;
// static size_t thompson_k = 3;
static double global_discount_rate = 0.99;
static double prior_discount_rate = 0.5;
static double local_discount_rate = 0.99;
static const stats::MovingStatistics::State roofline_prior({
    1.0,
    /*weight= */ 3.0,
    /*m2=     */ 3.0 * 1.0 / 9.0 // w * sigma ** 2
});
static const stats::MovingStatistics::State run_time_variation_prior({
    1.0,
    /*weight= */ 1.0,
    /*m2=     */ 1.0 * 1.0 / 16.0 // w * sigma ** 2
});

class GaussianBandit : public Bandit {
 public:
  GaussianBandit(
      selection::KernelEntryPoint::cost_estimates& costs,
      unsigned seed);
  api::Implementation choose() override;
  void update(api::Implementation choice, size_t delta_ns) override;

  // Global results. (Across keys)
  struct GlobalImplState {
    GlobalImplState() = default;
    GlobalImplState(const GlobalImplState&) = delete;

    stats::MovingStatistics roofline_correction{};
    stats::MovingStatistics run_time_variation{};
    size_t count{0};
  };

  // All data has the same key.
  struct LocalImplState {
    double roofline;
    stats::MovingStatistics measured{};
    size_t count{0};
    size_t global_count_at_last_update{0};
  };

 private:
  selection::KernelEntryPoint::supported_implementations implementations_;
  std::map<api::Implementation, std::unique_ptr<LocalImplState>>
      local_state_;
};
} // namespace bandits
} // namespace autotune
