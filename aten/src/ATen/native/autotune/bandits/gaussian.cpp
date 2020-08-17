#include <ATen/native/autotune/bandits/gaussian.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>

#include <ATen/native/autotune/api.h>
#include <ATen/native/autotune/bandits/common.h>
#include <ATen/native/autotune/dispatch/common.h>
#include <ATen/native/autotune/dispatch/core.h>
#include <ATen/native/autotune/utils/common.h>
#include <ATen/native/autotune/utils/logging.h>
#include <ATen/native/autotune/utils/stats.h>

namespace autotune {
namespace bandits {

GaussianBandit::GaussianBandit(
    selection::KernelEntryPoint::cost_estimates& costs,
    unsigned seed)
    : Bandit(costs, seed) {
  for (auto c : costs) {
    local_state_[c.impl] = std::make_unique<LocalImplState>();
    local_state_[c.impl]->roofline = c.cost;
    best_roofline_ = std::min({best_roofline_, c.cost});
  }
}

// TODO: this function is not thread safe.
api::Implementation GaussianBandit::choose() {
  auto choice = api::Implementation::kUnsupported;
  auto cost = std::numeric_limits<double>::max();
  int64_t local_count_sum = 0;

  // If the roofline predictions are overly pessimistic, (e.g. they neglect
  // vectorization), then the first implementation chosen appears much more
  // attractive than the rest. This causes the system prematurely specialize
  // on the one implementation that it has results for.
  double roofline_pessimism = 1.0;
  for (auto& l : local_state_) {
    auto& local_stats = l.second;
    local_count_sum += local_stats->count;
    if (local_stats->count) {
      auto pessimism = best_roofline_ / local_stats->measured.mean();
      roofline_pessimism = std::max({roofline_pessimism, pessimism});
    }
  }

  // Given the tendency of bandits to transition from explore-heavy
  // to exploit-heavy behavior, the presence of a ramp in the opposite
  // direction is curious. The reason is simple: an operation may only
  // be called a few times with a given configuration, and for such
  // dynamic workloads an early explore-heavy selection is little better
  // than a random bandit. So instead, the sampling rate ramps **down**
  // based on how many times a given configuration has been called
  // (regardless of which implementation was selected), and then if
  // a configuration is called many times it gracefully transitions to
  // the per-impl sampling rate ramp.
  int64_t min_k = std::max(thompson_k_max - local_count_sum / 2, (int64_t)1);

  for (auto& l : local_state_) {
    auto current_choice = l.first;
    auto& local_stats = l.second;

    auto local_count = local_stats->count;
    auto prior = roofline_prior * (local_stats->roofline / roofline_pessimism);
    auto measured = local_stats->measured.get_state();

    // https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_continuous_distribution
    auto alpha0 = prior.weight / 2.0;
    auto beta0 = prior.m2 / 2.0;
    auto nu = prior.weight + measured.weight;
    auto mean = (prior.weight * prior.mean + measured.weight * measured.mean) / nu;
    auto alpha = alpha0 + measured.weight / 2.0;
    auto beta_term2 = prior.weight * measured.weight / nu *
        std::pow(measured.mean - prior.mean, 2) / 2.0;
    auto beta = beta0 + 0.5 * measured.m2 + beta_term2;

    // What wikipedia calls "beta", C++ calls "1 / beta".
    auto inv_sigma_sq_dist = std::gamma_distribution<double>(alpha, 1.0 / beta);

    auto k = std::max(std::min(thompson_k_max, (int64_t)local_count + 1), min_k);
    double choice_cost = 0.0;
    for (size_t i = 0; i < k; i++) {
      auto stddev = std::sqrt(1.0 / inv_sigma_sq_dist(engine_) / nu);
      choice_cost += std::normal_distribution<double>(mean, stddev)(engine_);
    }

    if (choice_cost < cost) {
      choice = current_choice;
      cost = choice_cost;
    }
  }
  TORCH_INTERNAL_ASSERT(choice != api::Implementation::kUnsupported);
  return choice;
}

// TODO: this update is not thread safe.
void GaussianBandit::update(api::Implementation choice, size_t delta_ns) {
  stats::MovingStatistics::State sample((double)delta_ns * 1.0e-9);

  if (selection::DispatchInterface::singleton().times_chosen(choice) < warmup)
    // The first few times we see a kernel, the times tend to be wildly
    // high due to (presumably) lazy initialization. We use the global count
    // rather than just that of GaussianBandit, as these initializations are
    // not tied to a particular bandit.
    return;

  auto& local_stats = local_state_.at(choice);
  auto old_state = local_stats->measured.get_state();
  auto new_state = old_state.discount(local_discount_rate) + sample;
  local_stats->measured.set_state(new_state);
  local_stats->count++;
}

void GaussianBandit::summarize(selection::KernelEntryPoint::MapKey key) {
  printf("  %s\n", logging::to_string(key).c_str());
  for (auto i : implementations()) {
    auto& state = local_state_.at(i);
    auto count = state->count;
    auto mean = state->measured.mean();
    auto variance = (count > 1) ? state->measured.variance() : 0.0;
    printf(
      "  %-14s   %4d   %5.2f   %10.5f   %10.5f   %10.5f\n",
      logging::to_string(i).c_str(),
      (int)(count),
      mean / state->roofline,
      mean * 1.0e3,
      std::sqrt(variance) / mean,
      state->roofline
    );
  }
  printf("\n");
}

} // namespace bandits
} // namespace autotune
