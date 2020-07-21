#include <ATen/native/autotune/estimator.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <utility>

#include <ATen/native/autotune/definitions.h>

namespace autotune {

void StreamingVariance::update(double sample) {
  count_++;
  double m_old = m_;
  m_ += (sample - m_old) / count_;
  s_ += (sample - m_) * (sample - m_old);
}

double StreamingVariance::get() {
  return count_ > 1 ? s_ / (double)(count_ - 1) : 0.0;
}

Gaussian GaussianEstimatorBase::posterior() {
  // Based loosely on https://math.mit.edu/~dav/05.dir/class15-slides-all.pdf
  auto p = prior();
  auto ct = count();
  auto n = (double)ct;

  // This is a temporary hack. The posterior of a normal prior is only normal
  // if the sample variance is known, but this is intended to bootstrap out
  // of the low sample count regime. Very much still WIP.
  double variance = ct > 1 ? sample_variance_.get() : p.variance;

  double mean = ct > 0 ? total_ / ct : 0.0;

  double posterior_variance = 1.0 / (1.0 / p.variance + n / variance);
  double posterior_mean =
      posterior_variance * (p.mean / p.variance + n * mean / variance);

  return {posterior_mean, posterior_variance};
}

void GaussianEstimatorBase::update(double value) {
  sample_variance_.update(value);
  total_ += value;
}

std::ostream& operator<<(std::ostream & out, MovingPriorGaussianEstimator* e) {
    auto posterior = e->posterior();
    out << autotune::string_format(
        "M: %6.1f us,   S / M:  %6.2f,   Ct: %4d,   Corr:  %5.2f  (included)",
        posterior.mean * 1e6,
        std::sqrt(posterior.variance) / posterior.mean,
        e->count(),
        e->correction_->posterior().mean);
    return out;
}

} // namespace autotune
