#include <ATen/native/autotune/bandits/gaussian.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

#include <ATen/native/autotune/bandits/common.h>
#include <ATen/native/autotune/bandits/util.h>
#include <ATen/native/autotune/dispatch/common.h>
#include <ATen/native/autotune/dispatch/core.h>

namespace autotune {
namespace bandits {

using WorkingState = selection::DispatchWorkingState<GaussianBandit>;

GaussianBandit::GaussianBandit(
    selection::KernelEntryPoint::cost_estimates& costs,
    unsigned seed)
    : Bandit(costs, seed) {
  for (auto c : costs) {
    local_state_[c.impl] = std::make_unique<LocalImplState>();
    local_state_[c.impl]->roofline = c.cost;
  }
}

kernels::Implementation GaussianBandit::choose() {
  auto choice = kernels::Implementation::kUnsupported;
  auto cost = std::numeric_limits<double>::max();

  for (auto& l : local_state_) {
    auto current_choice = l.first;
    auto& local_state = l.second;
    auto global_stats = WorkingState::singleton().bandit_state(current_choice);

    auto distribution =
        (global_stats->roofline_correction.get_state() + roofline_prior) *
        local_state->roofline;

    auto local_count = local_state->count;
    auto prior_discount_factor = std::pow(prior_discount_rate, local_count);
    auto discount_prior = [prior_discount_factor](stats::MovingStatistics::State s) {
      return s.discount(prior_discount_factor);
    };

    if (local_count) {
      auto run_time_variation =
          (global_stats->run_time_variation.get_state() +
           run_time_variation_prior);

      auto m = local_state->measured.get_state();
      m = m + discount_prior(run_time_variation * m.mean);
      distribution = m + discount_prior(distribution);
    }

    double choice_cost = 0.0;
    for (size_t i = 0; i < thompson_k; i++) {
      choice_cost += stats::sample_normal(distribution, engine_);
    }

    if (choice_cost < cost) {
      choice = current_choice;
      cost = choice_cost;
    }
  }
  return choice;


  // // Temporarially randomly choose
  // std::uniform_int_distribution<size_t> distribution(
  //     0, implementations().size() - 1);
  // auto choice_index = distribution(engine_);
  // return implementations()[choice_index];
}

// TODO: this update is not thread safe.
void GaussianBandit::update(kernels::Implementation choice, size_t delta_ns) {
  stats::MovingStatistics::State sample((double)delta_ns * 1.0e-9);
  auto global_stats = WorkingState::singleton().bandit_state(choice);

  global_stats->count++;
  if (global_stats->count < warmup)
    //   The first few times we see a kernel, the times tend to be wildly
    //   high due to (presumably) lazy initialization.
    return;

  auto& local_stats = local_state_.at(choice);
  auto roofline = local_stats->roofline;
  auto old_state = local_stats->measured.get_state();
  auto new_state = old_state.discount(local_discount_rate) + sample;

  // Other kernels may have decayed our last contribution, so we need to
  // correct for that when updating this key's contribution
  auto update_correction = std::pow(
      global_discount_rate,
      global_stats->count - local_stats->global_count_at_last_update);

  auto roofline_correction =
      // Discount current state
      global_stats->roofline_correction.get_state().discount(
          global_discount_rate) +

      // Add the new contribution.
      (new_state * (1.0 / roofline) -

       // And remove the old.
       (old_state * (1.0 / roofline)).discount(update_correction));

  auto run_time_variation =
      // Discount current state
      global_stats->run_time_variation.get_state().discount(
          global_discount_rate) +

      // Add the new contribution.
      (new_state * (1.0 / new_state.mean) -

       // And remove the old.
       (old_state * (1.0 / (local_stats->count ? old_state.mean : 1)))
           .discount(update_correction));

  local_stats->measured.set_state(new_state);
  local_stats->count++;
  local_stats->global_count_at_last_update = global_stats->count;
  global_stats->roofline_correction.set_state(roofline_correction);
  global_stats->run_time_variation.set_state(run_time_variation);
}

} // namespace bandits
} // namespace autotune
