#include <ATen/native/autotune/estimator.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <utility>

#include <ATen/native/autotune/definitions.h>
#include <c10/util/Exception.h>

namespace stats {

void RunningStatistics::add_sample(RunningStatistics::State sample) {
  auto mean_old = mean();
  state_.weight_sum += sample.weight_sum;
  state_.mean += (sample.weight_sum / weight_sum()) * (sample.mean - mean_old);
  state_.s += sample.s +
      sample.weight_sum * (sample.mean - mean_old) * (sample.mean - mean());
}

void RunningStatistics::remove_sample(RunningStatistics::State sample) {
  TORCH_INTERNAL_ASSERT(weight_sum() - sample.weight_sum > 0.0);
  TORCH_INTERNAL_ASSERT(s() >= 0.0);
  auto mean_old = mean();

  state_.mean = (mean_old - sample.weight_sum / weight_sum() * sample.mean ) / (1.0 - sample.weight_sum / weight_sum());
  state_.s -= sample.s + sample.weight_sum * (sample.mean - mean_old) * (sample.mean - mean());
  state_.weight_sum -= sample.weight_sum;

  TORCH_INTERNAL_ASSERT(state_.s >= 0.0);
}

void RunningStatistics::decay(double factor) {
  state_ = discount(state_, factor);
}

double RunningStatistics::mean() {
  return state_.mean;
};

double RunningStatistics::variance() {
  TORCH_INTERNAL_ASSERT(weight_sum() > 1.0);
  TORCH_INTERNAL_ASSERT(s() >= 0);
  return s() / (weight_sum() - 1.0);
};

double RunningStatistics::weight_sum() {
  TORCH_INTERNAL_ASSERT(state_.weight_sum >= 0.0);
  return state_.weight_sum;
};

double RunningStatistics::s() {
  TORCH_INTERNAL_ASSERT(state_.s >= 0.0);
  return state_.s;
};

RunningStatistics::State scale(RunningStatistics::State rs, double factor) {
  return {rs.mean * factor, rs.weight_sum, rs.s * std::pow(factor, 2)};
}

RunningStatistics::State discount(RunningStatistics::State rs, double factor) {
  return {rs.mean, rs.weight_sum * factor, rs.s * factor};
}

RunningStatistics::State merge(RunningStatistics::State first, RunningStatistics::State second) {
  RunningStatistics out {first};
  out.add_sample(second);
  return out;
}

double sample_normal(RunningStatistics::State s, std::mt19937& engine) {
  RunningStatistics rs {s};
  return std::normal_distribution<double>(rs.mean(), std::sqrt(rs.variance()))(engine);
}

std::ostream& operator<<(std::ostream & out, RunningStatistics& s) {
  auto v = s.weight_sum() > 1 ? s.variance() : 0.0;
  auto m = s.mean();
  auto s_over_m = m > 0 ? std::sqrt(v) / m : 0.0;
  out << autotune::string_format("M: %12.10f   %5.3f", m, s_over_m);
  return out;
}

std::ostream& operator<<(std::ostream& out, RunningStatistics::State s) {
  RunningStatistics rs (s);
  out << rs;
  return out;
}

} // namespace stats
