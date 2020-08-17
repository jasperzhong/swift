#pragma once

#include <algorithm>
#include <limits>
#include <map>

#include <ATen/native/autotune/api.h>
#include <ATen/native/autotune/bandits/common.h>
#include <ATen/native/autotune/dispatch/common.h>
#include <ATen/native/autotune/utils/stats.h>

namespace autotune {
namespace bandits {

static size_t warmup = 50;
static int64_t thompson_k_max = 4;
static double local_discount_rate = 0.99;
static const stats::MovingStatistics::State roofline_prior({
    /*mean=   */ 1.0,
    /*weight= */ 2.0,
    /*m2=     */ 2.0 * 1.0 / 16.0 // w * sigma ** 2
});

class GaussianBandit : public Bandit {
 public:
  GaussianBandit(
      selection::KernelEntryPoint::cost_estimates& costs,
      unsigned seed);
  api::Implementation choose() override;
  void update(api::Implementation choice, size_t delta_ns) override;
  void summarize(selection::KernelEntryPoint::MapKey) override;

 private:
  struct LocalImplState {
    double roofline;
    stats::MovingStatistics measured{};
    size_t count{0};
  };

  std::map<api::Implementation, std::unique_ptr<LocalImplState>> local_state_;
  double best_roofline_ = std::numeric_limits<double>::max();
};
} // namespace bandits
} // namespace autotune
