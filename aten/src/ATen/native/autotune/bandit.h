#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <ATen/native/autotune/definitions.h>
#include <ATen/native/autotune/estimator.h>

namespace bandits {

static size_t thompson_k = 1;
using gaussian_bandit_estimates = std::map<
    autotune::DispatchChoice,
    std::unique_ptr<autotune::MovingPriorGaussianEstimator>>;

class GaussianBandit {
 public:
  GaussianBandit(gaussian_bandit_estimates& priors, uint64_t seed)
      : estimates_(std::move(priors)), engine_(seed){};
  autotune::DispatchChoice select();
  void update(autotune::DispatchChoice choice, double value);

 private:
  gaussian_bandit_estimates estimates_;
  std::mt19937 engine_;
};

} // namespace bandits
