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








struct MeanVariance {
  double mean;
  double variance;
};

static MeanVariance merge_normal(MeanVariance d0, MeanVariance d1) {
  auto v = 1.0 / (1.0 / d0.variance + 1.0 / v1.variance);
  auto m = (d0.mean * d1.variance + d1.mean * d0.variance);
  return {m, v};
}

static double sample_normal(MeanVariance d, std::mt19937& engine) {
  return std::normal_distribution<double>(d.mean, std::sqrt(d.variance))(engine);
}

constexpr double default_forgetfulness = 0.4;
class RunningMeanVariance {
 public:
  RunningMeanVariance(
      double prior_mean,
      size_t prior_count,
      double forgetfulness = default_forgetfulness)
      : m_(prior_mean),
        count_(prior_count),
        prior_mean_(prior_mean),
        prior_count_(prior_count),
        forgetfulness_(forgetfulness){};
  RunningMeanVariance() : RunningMeanVariance(0, 0, 0){};

  void update(double sample);
  MeanVariance get();

 private:
  // Underlying streaming variance equations:
  //   https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
  //   https://www.johndcook.com/blog/standard_deviation/
  double m_;
  double s_ = 0;
  size_t count_ = 0;

  void add_sample(double sample);
  void remove_sample(double sample);

  double prior_mean_;
  size_t prior_count_;
  double forgetfulness_;
  void maybe_forget_prior(double sample);

  friend std::ostream& operator<<(std::ostream & out, RunningMeanVariance r);
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
