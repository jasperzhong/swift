#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>

namespace autotune {

struct Gaussian {
  Gaussian(double m, double v) : mean(m), variance(v){};
  double mean;
  double variance;
  double sample(std::mt19937& engine) {
    return std::normal_distribution<double>(mean, std::sqrt(variance))(engine);
  }
};

class StreamingVariance {
 public:
  void update(double sample);
  double get();
  size_t count() {
    return count_;
  };

 private:
  // https://www.johndcook.com/blog/standard_deviation/
  double m_ = 0;
  double s_ = 0;
  size_t count_ = 0;
};

class GaussianEstimatorBase {
 public:
  virtual Gaussian prior() = 0;
  Gaussian posterior();
  void update(double value);
  size_t count() { return sample_variance_.count(); };

 private:
  StreamingVariance sample_variance_;
  double total_{0};
};

class GaussianEstimator : public GaussianEstimatorBase {
 public:
  GaussianEstimator(double mean, double variance)
      : mean_(mean), variance_(variance){};
  Gaussian prior() {
    return {mean_, variance_};
  };

 private:
  double mean_;
  double variance_;
};

class MovingPriorGaussianEstimator : public GaussianEstimatorBase {
 public:
  MovingPriorGaussianEstimator(
      double mean,
      double variance,
      std::shared_ptr<GaussianEstimator> correction)
      : mean_(mean), variance_(variance), correction_(correction){};

  Gaussian prior() {
    auto c = correction_->posterior().mean;
    return {c * mean_, std::pow(c, 2) * variance_};
  };

 private:
  double mean_;
  double variance_;
  std::shared_ptr<GaussianEstimator> correction_;

  friend std::ostream& operator<<(std::ostream & out, MovingPriorGaussianEstimator* e);
};

} // namespace autotune
