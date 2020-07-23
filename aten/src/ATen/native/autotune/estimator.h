#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include <ATen/native/autotune/definitions.h>
#include <c10/util/Exception.h>

namespace stats {

// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
class RunningStatistics {
 public:
  struct State;
  RunningStatistics() : RunningStatistics(State(0.0, 0.0, 0.0)){};
  RunningStatistics(State s) : state_(s){};
  RunningStatistics(const RunningStatistics&) = delete;
  RunningStatistics(RunningStatistics&& other)
      : RunningStatistics(other.state_){};

  void add_sample(State sample);
  void remove_sample(State other);
  void decay(double factor);

  double mean();
  double variance();

  struct State {
    State(double m, double w = 1.0, double s0 = 0.0)
        : mean(m), weight_sum(w), s(s0){};
    State(RunningStatistics& st) : State(st.state_) {};
    double mean;
    double weight_sum;
    double s;
  };

 private:
  friend struct State;
  State state_;
  double weight_sum();
  double s();

  friend std::ostream& operator<<(std::ostream& out, RunningStatistics& s);
};

std::ostream& operator<<(std::ostream& out, RunningStatistics::State s);
RunningStatistics::State scale(RunningStatistics::State rs, double factor);
RunningStatistics::State discount(RunningStatistics::State rs, double factor);
RunningStatistics::State merge(RunningStatistics::State first, RunningStatistics::State second);
double sample_normal(RunningStatistics::State s, std::mt19937& engine);
} // namespace stats


namespace autotune {
struct ImplStats {
  ImplStats() = default;
  ImplStats(const ImplStats&) = delete;

  stats::RunningStatistics roofline_correction {};
  stats::RunningStatistics run_time_variation {};
  size_t count;
};

std::shared_ptr<ImplStats> get_impl_stats(DispatchChoice choice);
} // namespace autotune
