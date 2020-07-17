#include <ATen/native/autotune/bandit.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>


namespace bandits {

GaussianBandit::GaussianBandit(autotune::cost_priors priors, uint64_t seed)
    : engine_(seed) {
  for (auto p : priors){
    auto choice = p.first;
    auto mean = p.second;

    // TODO: This is a placeholder until backend
    //       specific priors are integrated.
    auto variance = std::pow(mean, 2) / 9.0;

    distributions_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(choice),
        std::forward_as_tuple(mean, variance));
  }
}

autotune::DispatchChoice GaussianBandit::select() {
    auto choice = autotune::DispatchChoice::kUnsupported;
    auto cost = std::numeric_limits<double>::max();

    for (auto& d : distributions_) {
      auto mv = d.second.get(); // Current posterior mean and variance;
      std::normal_distribution<double> generator(
          /*mean=*/mv.first, /*stddev=*/std::sqrt(mv.second));

      double choice_cost = 0.0;
      for (size_t i = 0; i < thompson_k; i++)
        choice_cost += generator(engine_);

      if (choice_cost < cost) {
          choice = d.first;
          cost = choice_cost;
      }
    }

    return choice;
}

void GaussianBandit::update(autotune::DispatchChoice choice, double value) {
    auto& d = distributions_.at(choice);
    std::cout << "  " << d << std::endl;

    d.update(value);
    std::cout << "  " << d << std::endl;
}



















Bandit::Bandit(std::vector<double> priors, uint64_t seed, std::string repr)
    : repr_(repr), n_(priors.size()), mu_(n_), counts_(n_) {
  // TODO: learned global roofline correction priors.
  upper_bound_ = std::accumulate(priors.begin(), priors.end(), 0.0);
  initial_upper_bound_ = upper_bound_;
  for (int i = 0; i < n_; i++) {
    mu_[i] = priors[i];
  }

  engine_ = std::mt19937(seed);
}

void Bandit::print() {
    int th_count = counts_[0];
    int mkl_count = counts_[1];
    auto total_count = th_count + mkl_count;
    if (!(total_count == 5 || total_count == 25 || total_count == 100 || total_count == 500)) return;

    printf("TH chosen %d / %d, (%5.1f%%)\n", th_count, total_count, (double)th_count / (double)total_count * 100);
    printf("  TH mean:  %9.1f us   (includes prior and damping)\n", mu_[0] / (double)th_count * 1e6);
    printf("  MKL mean: %9.1f us   (includes prior and damping)\n", mu_[1] / (double)mkl_count * 1e6);
    printf("  Max value: %5.1f us\n", upper_bound_ * 1e6);
    std::cout << repr_ << std::endl << std::endl;
}


void Bandit::Callback::end(){
    auto t1 = std::chrono::high_resolution_clock::now();
    auto delta_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0_).count();
    double delta_sec = (double)delta_ns * 1.0e-9;

    // Damp values which are well out of bounds to
    // prevent early lock in as the system warms up.
    // TODO: replace this heuristic.
    delta_sec = delta_sec > 2. * b_->upper_bound_ ? 2. * b_->upper_bound_ : delta_sec;

    if (delta_sec > b_->upper_bound_) b_->upper_bound_ = delta_sec;
    b_->mu_[choice_] += delta_sec;
    b_->counts_[choice_] += 1;

    b_->print();
}


Bandit::Callback Bandit::sample() {
    size_t choice = 0;
    double current_best = (double)(thompson_k + 1);

    for (size_t candidate = 0; candidate < n_; candidate ++) {
        double candidate_total = 0.0;
        auto ct = counts_[candidate];
        auto mu = mu_[candidate] / upper_bound_ / (double)(ct + 1);
        double counts_plus_two = (double)(ct + 2);
        double a = mu * counts_plus_two;
        double b = (1. - mu) * counts_plus_two;

        std::gamma_distribution<double> distribution_0(a, 1.);
        std::gamma_distribution<double> distribution_1(b, 1.);
        for (size_t i = 0; i < thompson_k; i++){
            auto x0 = distribution_0(engine_);
            auto x1 = distribution_1(engine_);
            candidate_total += x0 / (x0 + x1);
        }
        if (candidate_total < current_best) {
            current_best = candidate_total;
            choice = candidate;
        }
    }

    return Callback(this, choice);
}

}  // bandits
