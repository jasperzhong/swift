#pragma once

#include <algorithm>
#include <map>

#include <ATen/native/autotune/api.h>
#include <ATen/native/autotune/bandits/common.h>
#include <ATen/native/autotune/dispatch/common.h>
#include <ATen/native/autotune/utils/stats.h>

namespace autotune {
namespace bandits {

static double variance_clip = 1.0 / 7.0;
static int64_t thompson_k(size_t count) {
  int64_t max_k = 4;
  return (count + 1 < max_k) ? count + 1 : max_k;
}

static size_t warmup = 50;
static double prior_discount_rate = 0.75;
static double local_discount_rate = 0.99;
static const stats::MovingStatistics::State roofline_prior({
    1.0,
    /*weight= */ 2.0,
    /*m2=     */ 2.0 * 1.0 / 4.0 // w * sigma ** 2
});

class GaussianBandit : public Bandit {
 public:
  GaussianBandit(
      selection::KernelEntryPoint::cost_estimates& costs,
      unsigned seed);
  api::Implementation choose() override;
  void update(api::Implementation choice, size_t delta_ns) override;
  void summarize(selection::KernelEntryPoint::MapKey) override;

  struct LocalImplState {
    double roofline;
    stats::MovingStatistics measured{};
    size_t count{0};
  };

 private:
  selection::KernelEntryPoint::supported_implementations implementations_;
  std::map<api::Implementation, std::unique_ptr<LocalImplState>>
      local_state_;
};
} // namespace bandits
} // namespace autotune
