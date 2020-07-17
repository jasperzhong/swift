#include <ATen/native/autotune/estimator.h>

#include <cmath>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <utility>

#include <ATen/native/autotune/definitions.h>

namespace autotune {

GaussianEstimator::GaussianEstimator(double prior_mean, double prior_variance)
    : prior_mean_(prior_mean), prior_variance_(prior_variance) {}

std::ostream& operator<<(std::ostream & out, GaussianEstimator& e){
    auto mv = e.get();
    out << autotune::string_format(
        "Mean: %9.1f us,  Std: %9.1f us",
        mv.first * 1e6, std::sqrt(mv.second) * 1e6);
    return out;
}

void GaussianEstimator::update(double sample){
    total_ += sample;
    count_++;

    double m_old = variance_m_;

    variance_m_ += (sample - m_old) / count_;
    variance_s_ += (sample - variance_m_) * (sample - m_old);
}

double GaussianEstimator::sample_variance() {
  return count_ > 1 ? variance_s_ / (double)(count_ - 1) : 0.0;
}

std::pair<double, double> GaussianEstimator::get() {
  // Based loosely on https://math.mit.edu/~dav/05.dir/class15-slides-all.pdf
  auto count = (double)count_;

  // This is a hack. The posterior of a normal prior is only normal if the
  // sample variance is known, but as a practical matter we substitute the
  // sample variance since it only matters when bootstrapping.
  double variance = count_ > 1 ? sample_variance() : prior_variance_;
  double mean = count_ > 0 ? total_ / count : 0.0;

  double posterior_variance =
      1.0 / (1.0 / prior_variance_ + count / variance);
  double posterior_mean = posterior_variance *
      (prior_mean_ / prior_variance_ + count * mean / variance);

  return {posterior_mean, posterior_variance};
}

} // namespace autotune
