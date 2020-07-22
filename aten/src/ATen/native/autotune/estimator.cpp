#include <ATen/native/autotune/estimator.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <utility>

#include <ATen/native/autotune/definitions.h>

namespace autotune {





void RunningMeanVariance::update(double sample) {
  maybe_forget_prior(sample);
  add_sample(sample);
}


void RunningMeanVariance::add_sample(double sample) {
  count_++;
  auto m_old = m_;
  m_ += (sample - m_old) / (double)count_;
  s_ += (sample - m_) * (sample - m_old);
}

void RunningMeanVariance::remove_sample(double sample) {
  if (count_ < 2) {
    m_ = 0;
    s_ = 0;
    count_ = 0;
    return;
  }

  auto m_kminus1 = ((double)count_ * m_ - sample) / (double)(count_ - 1);
  s_ -= (sample - m_) * (sample - m_kminus1);
  s_ = std::max(s_, 0.0); // Prevent underflow.
  m_ = m_kminus1;
  count_--;
}

void RunningMeanVariance::maybe_forget_prior(double sample) {
  if (!prior_count_)
    return;

  auto current = get();
  auto forget_range = std::sqrt(current.variance) * forgetfulness_;
  auto lower_bound = current.mean - forget_range;
  auto upper_bound = current.mean + forget_range;
  if (sample >= lower_bound && sample <= upper_bound) {
    prior_count_--;
    remove_sample(prior_mean_);
  }
}

MeanVariance RunningMeanVariance::get() {
  return {m_, (count_ > 1) ? s_ / (double)(count_ - 1) : 0.0};
}

std::ostream& operator<<(std::ostream & out, RunningMeanVariance r) {
  auto mv = r.get();
  out << autotune::string_format(
    "%6.2f   %6.2f    %d", mv.mean, std::sqrt(mv.variance), r.count_
  );
  return out;
}














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
