#pragma once

#include <cstddef>
#include <iostream>
#include <utility>

namespace autotune {
class GaussianEstimator {
 public:
  GaussianEstimator(double prior_mean, double prior_variance);
  void update(double sample);

  // Posterior mean, variance.
  std::pair<double, double> get();

 private:
  double prior_mean_;
  double prior_variance_;

  double total_ = 0;
  size_t count_ = 0;

  // https://www.johndcook.com/blog/standard_deviation/
  double variance_m_ = 0;
  double variance_s_ = 0;
  double sample_variance();

  friend std::ostream& operator<<(std::ostream&, GaussianEstimator&);
};

} // namespace autotune
