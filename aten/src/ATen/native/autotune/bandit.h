#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <ATen/native/autotune/definitions.h>
#include <ATen/native/autotune/estimator.h>

namespace bandits {

static double global_decay_rate = 0.99;
static double prior_decay_rate = 0.9;
static double local_decay_rate = 0.99;

static size_t thompson_k = 2;
static const stats::RunningStatistics::State roofline_prior({
    1.0,
    /*weight= */ 3.0,
    /*s=      */ 3.0 * 1.0 / 9.0 // w * sigma ** 2
});
static const stats::RunningStatistics::State run_time_variation_prior({
    1.0,
    /*weight= */ 3.0,
    /*s=      */ 3.0 * 1.0 / 9.0 // w * sigma ** 2
});

static double gamma = 0.5772156649; // Euler–Mascheroni constant
static double harmonic(double i) {
    return std::log(i) + gamma + 0.5 / i - 1.0 / 12.0 / std::pow(i, 2);
}

struct Results {
  Results(double r) : roofline(r){};
  Results(const Results&) = delete;

  double roofline;
  stats::RunningStatistics measured{};
  size_t count {0};

  stats::RunningStatistics::State roofline_correction(bool discount);
  stats::RunningStatistics::State run_to_run_variation(bool discount);

 private:
  stats::RunningStatistics::State derived_distribution(double scale_factor, bool discount);
};

using gaussian_bandit_results = std::map<
    autotune::DispatchChoice,
    std::unique_ptr<Results>>;

class GaussianBandit {
 public:
  GaussianBandit(gaussian_bandit_results& priors, uint64_t seed)
      : impl_results_(std::move(priors)), engine_(seed){};
  autotune::DispatchChoice select();
  void update(autotune::DispatchChoice choice, double value);

 private:
  gaussian_bandit_results impl_results_;
  std::mt19937 engine_;
};

} // namespace bandits
