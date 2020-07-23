#include <ATen/native/autotune/bandit.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>


namespace bandits {

stats::RunningStatistics::State Results::derived_distribution(
    double scale_factor,
    bool discount) {
  if (!count)
    return {0.0, 0.0, 0.0};

  auto scaled = stats::scale(measured, scale_factor);
  if (count == 1 || !discount)
    return scaled;

  auto discount_factor = harmonic((double)count) / (double)count;
  return stats::discount(scaled, discount_factor);
};

stats::RunningStatistics::State Results::roofline_correction(bool discount) {
  return derived_distribution(1.0 / roofline, discount);
}

stats::RunningStatistics::State Results::run_to_run_variation(bool discount) {
  return derived_distribution(1.0 / measured.mean(), discount);
}

autotune::DispatchChoice GaussianBandit::select() {
  auto choice = autotune::DispatchChoice::kUnsupported;
  auto cost = std::numeric_limits<double>::max();

  for (auto& r : impl_results_) {
    auto current_choice = r.first;
    auto& local_results = r.second;
    auto global_stats = autotune::get_impl_stats(current_choice);

    auto distribution = stats::scale(
        stats::merge(global_stats->roofline_correction, roofline_prior),
        local_results->roofline);

    auto local_count = local_results->count;
    auto prior_discount_factor = std::pow(prior_decay_rate, local_count);
    auto discount_prior = [prior_discount_factor](stats::RunningStatistics::State s) {
      return stats::discount(s, prior_discount_factor);
    };
    if (local_count) {
      auto run_time_variation = stats::merge(
          global_stats->run_time_variation,
          run_time_variation_prior);

      // Clone measured state.
      stats::RunningStatistics::State sample_distribution = stats::merge(
          local_results->measured,
          stats::scale(discount_prior(run_time_variation), sample_distribution.mean));
      distribution = stats::merge(
        discount_prior(distribution),
        sample_distribution);
    }

    double choice_cost = 0.0;
    for (size_t i = 0; i < thompson_k; i++) {
      choice_cost += stats::sample_normal(distribution, engine_);
    }

    std::cout << "Candidate:       " << distribution << "  " << choice_cost / distribution.mean << std::endl;

    if (choice_cost < cost) {
      choice = r.first;
      cost = choice_cost;
    }
  }

  return choice;
}

void GaussianBandit::update(
    autotune::DispatchChoice choice,
    double value) {

  auto global_stats = autotune::get_impl_stats(choice);
  global_stats->count++;
  if (global_stats->count <  50)
    // TODO: Formalize
    //   The first few times we see a kernel, the times tend to be wildly
    //   high due to (presumably) lazy initialization.
    return;

  auto& impl_result = impl_results_.at(choice);
  auto old_roofline_correction =
      impl_result->roofline_correction(/*discount=*/true);
  auto old_run_to_run =
      impl_result->run_to_run_variation(/*discount=*/true);

  impl_result->measured.decay(local_decay_rate);
  impl_result->measured.add_sample({value});
  impl_result->count++;

  global_stats->roofline_correction.add_sample(
      impl_result->roofline_correction(/*discount=*/true));
  global_stats->roofline_correction.remove_sample(old_roofline_correction);

  global_stats->run_time_variation.add_sample(
      impl_result->run_to_run_variation(/*discount=*/true));
  global_stats->run_time_variation.remove_sample(old_run_to_run);

  // std::cout << global_stats->roofline_correction << std::endl << std::endl;
  // std::cout << global_stats->run_time_variation << std::endl << std::endl;
  std::cout << impl_result->measured << std::endl;
  std::cout << "  ";
  for (auto& r : impl_results_) {
    std::cout << r.second->count << "  ";
  }
  std::cout << std::endl;
}

} // namespace bandits
