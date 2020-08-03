#include <ATen/native/autotune/utils/stats.h>

#include <cmath>
#include <iostream>
#include <random>

#include <c10/util/Exception.h>

namespace autotune {
namespace stats {

using State = MovingStatistics::State;

void validate(const State& s) {
  TORCH_INTERNAL_ASSERT(s.weight >= 0.0);
  TORCH_INTERNAL_ASSERT(s.m2 >= 0.0, "M2 has value: ", s.m2, " < 0.");
}

void assert_close(const State& first, const State& second) {
  auto close = [](double x, double y) {
    double epsilon = 1.0e-10;
    auto diff = std::abs(x - y);
    return (diff < epsilon) || (2.0 * diff) / (x + y) < epsilon;
  };
  TORCH_INTERNAL_ASSERT(close(first.mean, second.mean));
  TORCH_INTERNAL_ASSERT(close(first.weight, second.weight));
  TORCH_INTERNAL_ASSERT(close(first.m2, second.m2));
}

State MovingStatistics::get_state() {
  return state_;
}

void MovingStatistics::set_state(const State new_state) {
  validate(new_state);
  state_ = new_state;
}

void MovingStatistics::add(State sample) {
  validate(sample);

  // Extra safety. This can be removed once the numerics prove
  // stable through a sufficient amount of testing.
  assert_close(state_, State(state_ + sample) - sample);

  state_ = state_ + sample;
  validate(state_);
}

void MovingStatistics::remove(State sample) {
  validate(sample);

  // Extra safety. This can be removed once the numerics prove
  // stable through a sufficient amount of testing.
  assert_close(state_, State(state_ - sample) + sample);

  state_ = state_ - sample;
  validate(state_);
}

double MovingStatistics::mean() {
    return state_.mean;
}

double MovingStatistics::variance(bool run_checks) {
    if (run_checks) {
        TORCH_INTERNAL_ASSERT(state_.weight > 1.0);
        TORCH_INTERNAL_ASSERT(state_.m2 > 0.0);
    }
    return state_.m2 / (state_.weight - 1.0);
}

State State::operator+(State other) {
  auto weight_new = weight + other.weight;
  auto delta = other.mean - mean;
  auto mean_new = mean + (other.weight / weight_new) * delta;
  auto m2_new =
      m2 + other.m2 + std::pow(delta, 2) * weight * other.weight / weight_new;

  return {mean_new, weight_new, m2_new};
}

State State::operator-(State other) {
  auto mean_new = (mean - other.weight / weight * other.mean) /
      (1.0 - other.weight / weight);
  auto m2_new = m2 - other.m2 -
      other.weight * (other.mean - mean) * (other.mean - mean_new);
  return {mean_new, weight - other.weight, m2_new};
}

State State::operator*(double factor) {
  return {mean * factor, weight, m2 * std::pow(factor, 2)};
}

State State::discount(double factor) {
    return {mean, weight * factor, m2 * factor};
}

double sample_normal(State state, std::mt19937& engine, int64_t n) {
  MovingStatistics s {state};
  double stddev = std::sqrt(s.variance() / (double)n);
  return std::normal_distribution<double>(s.mean(), stddev)(engine);
}

} // namespace stats
} // namespace autotune
