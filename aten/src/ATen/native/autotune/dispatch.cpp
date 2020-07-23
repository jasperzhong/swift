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

std::shared_ptr<ImplStats> CostDispatcher::get_impl_stats(autotune::DispatchChoice choice) {
  return impl_stats_[static_cast<size_t>(choice)];
}

std::shared_ptr<ImplStats> get_impl_stats(DispatchChoice choice) {
  return CostDispatcher::singleton().get_impl_stats(choice);
}

CostDispatcher::RAII_Choice::RAII_Choice(DispatchChoice choice, bandit_ptr b)
  : choice_(choice), b_(b) {
  start_ = std::chrono::high_resolution_clock::now();
}


void CostDispatcher::RAII_Choice::finished() {
  auto end = std::chrono::high_resolution_clock::now();
  if (b_ == nullptr)
    return;

  auto delta_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
  std::cout << autotune::string_format(
                   "~RAII_Choice: %d   %6.1f us",
                   static_cast<size_t>(choice_),
                   (double)delta_ns / 1.0e3)
            << std::endl;

  b_->update(choice_, (double)delta_ns / 1.0e9);
}

// CostDispatcher::RAII_Choice::~RAII_Choice(){
//   finished();
// }



CostDispatcher::RAII_Choice CostDispatcher::choose(EntryPoint e){
  auto key = e.key();
  if (std::get<0>(key) == DispatchGroup::kNotApplicable)
    return {e.value()[0].impl, nullptr};

  if (bandits_.find(key) == bandits_.end()){
    bandits::gaussian_bandit_results priors;
    for (auto implementation_prior : e.value()) {
      auto roofline = implementation_prior.cost;
      priors[implementation_prior.impl] =
          std::make_unique<bandits::Results>(roofline);
    }

    bandits_[key] = std::make_shared<bandits::GaussianBandit>(priors, next_seed_);
    next_seed_++;
  }

  auto bandit = bandits_.at(key);
  auto choice = bandit->select();

  return {choice, bandit};
}

}  // autotune
