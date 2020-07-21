#include <ATen/native/autotune/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <c10/util/ArrayRef.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/native/autotune/bandit.h>
#include <ATen/native/autotune/cost.h>


namespace autotune {

CostDispatcher::RAII_Choice::RAII_Choice(EntryPoint::map_key key, DispatchChoice choice, bandit_ptr b)
  : key_(key), choice_(choice), b_(b) {
  start_ = std::chrono::high_resolution_clock::now();
}

CostDispatcher::RAII_Choice::~RAII_Choice(){
  auto end = std::chrono::high_resolution_clock::now();
  if (b_ == nullptr)
    return;

  auto delta_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();

  //TODO: remove `key_` once I remove this print line.
  std::cout << "~RAII_Choice: "
            << static_cast<size_t>(std::get<0>(key_)) << "  "
            << std::get<1>(key_) << "  "
            << std::get<2>(key_) << "  "
            << static_cast<size_t>(choice_) << "  "
            << delta_ns / 1000 << " us" << std::endl;

  b_->update(choice_, (double)delta_ns / 1.0e9);
}

CostDispatcher::RAII_Choice CostDispatcher::choose(EntryPoint e){
  auto key = e.key();
  if (std::get<0>(key) == DispatchGroup::kNotApplicable)
    return {key, e.value()[0].impl, nullptr};

  if (bandits_.find(key) == bandits_.end()){
    bandits::gaussian_bandit_estimates priors;
    for (auto implementation_prior : e.value()) {
      auto choice = implementation_prior.impl;
      auto choice_as_index = static_cast<size_t>(choice);
      auto cost = implementation_prior.cost;

      // TODO: This is a placeholder until a better
      //       heuristic is chosen.
      auto variance = std::pow(cost, 2) / 9.0;

      priors[choice] = std::make_unique<autotune::MovingPriorGaussianEstimator>(
        cost, variance, prior_correction_[choice_as_index]);
    }

    bandits_[key] = std::make_shared<bandits::GaussianBandit>(priors, next_seed_);
    next_seed_++;
  }

  auto bandit = bandits_.at(key);
  auto choice = bandit->select();

  return {key, choice, bandit};
}

}  // autotune
