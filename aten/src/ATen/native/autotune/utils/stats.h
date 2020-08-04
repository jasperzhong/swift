#pragma once

#include <random>

namespace autotune {
namespace stats {

/*
This class keeps a running (weighted) mean and variance using Welford's
algorithm.
(https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)

Because of its mutable nature, the copy constructor is deleted to avoid bugs
where a one updates a copy rather than the canonical tally. Instead, all
counters are stored in a `State` struct so that explicit copies are still
possible.

Several higher order transformations are also provided, such as merging or
scaling running statistics. All of these transformations operate on
MovingStatistics::State objects. Once the transformation(s) are complete
the result can be used to construct a new MovingStatistics object. This is
done for both a logical reason (to reduce the cognitive overhead of std::move's)
and a performance one (`State` is three doubles and many of the transforms are
simple arithmetic operations, so significant copy elision and inlining is
expected unless we interfere).
*/
class MovingStatistics {
 public:
  struct State;
  // NB: For the empty constructor, we do not use the default values of `State`
  // because an empty set of moving statistics should have weight=0.
  MovingStatistics() : MovingStatistics(State(0.0, 0.0, 0.0)){};
  MovingStatistics(State s) : state_(s){};
  MovingStatistics(const MovingStatistics&) = delete;
  MovingStatistics(MovingStatistics&& other) : MovingStatistics(other.state_){};

  State get_state();
  void set_state(const State new_state);

  void add(State sample);
  void remove(State sample);

  double mean();
  double variance(bool run_checks = true);

  struct State {
    State(double m, double w = 1.0, double m2_contribution = 0.0)
        : mean(m), weight(w), m2(m2_contribution){};
    double mean;
    double weight;
    double m2; // Unscaled second moment.

    // These methods do not enforce weight >=0 and m2 >= 0
    // MovingStatistics::add and MovingStatistics::remove do.
    State operator+(State);
    State operator-(State);
    State operator*(double);
    State discount(double);
  };

 private:
  State state_;
};

double sample_normal(MovingStatistics::State s, std::mt19937& engine, int64_t n = 1);
double sample_normal(double mean, double variance, std::mt19937& engine, int64_t n = 1);

} // namespace stats
} // namespace autotune
