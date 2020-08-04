#include <ATen/native/autotune/bandits/gaussian.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
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
  }
}

// TODO: this update is not thread safe.
api::Implementation GaussianBandit::choose() {
  auto choice = api::Implementation::kUnsupported;
  auto cost = std::numeric_limits<double>::max();

  double roofline_shift = 1.0;
  for (auto& l : local_state_) {
    auto& local_stats = l.second;
    auto measured = local_stats->measured.get_state();
    auto roofline = local_stats->roofline;
    if (measured.weight >= 1.0) {
      roofline_shift = std::max({roofline_shift, roofline / measured.mean});
    }
  }

  for (auto& l : local_state_) {
    auto current_choice = l.first;
    auto& local_stats = l.second;

    auto local_count = local_stats->count;
    auto prior = roofline_prior;
    auto prior_discount = std::pow(prior_discount_rate, local_count);
    auto distribution =
        (prior.discount(prior_discount) * (local_stats->roofline / roofline_shift));

    if (local_count) {
      distribution = distribution + local_stats->measured.get_state();
    }

    // double choice_cost =
    //     stats::sample_normal(distribution, engine_, thompson_k(local_count));

    stats::MovingStatistics d = distribution;
    // auto variance = std::min({d.variance(), variance_clip});
    auto variance = d.variance() / distribution.weight;
    double choice_cost =
        stats::sample_normal(d.mean(), variance, engine_, thompson_k(local_count));

    if (choice_cost < cost) {
      choice = current_choice;
      cost = choice_cost;
    }
  }
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
      "  %-14s   %4d   %5.2f   %10.5f   %10.5f\n",
      logging::to_string(i).c_str(),
      (int)(count),
      mean / state->roofline,
      mean * 1.0e3,
      std::sqrt(variance) / mean
    );
  }
  printf("\n");
}

} // namespace bandits
} // namespace autotune
