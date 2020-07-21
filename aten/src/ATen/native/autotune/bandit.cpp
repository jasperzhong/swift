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

autotune::DispatchChoice GaussianBandit::select() {
  auto choice = autotune::DispatchChoice::kUnsupported;
  auto cost = std::numeric_limits<double>::max();

  for (auto& e : estimates_) {
    auto current_posterior = e.second->posterior();

    double choice_cost = 0.0;
    for (size_t i = 0; i < thompson_k; i++)
      choice_cost += current_posterior.sample(engine_);

    if (choice_cost < cost) {
      choice = e.first;
      cost = choice_cost;
    }
  }

  return choice;
}

void GaussianBandit::update(autotune::DispatchChoice choice, double value) {
  for (auto& i : estimates_) {
      std::cout << (i.first == choice ? " > " : "   ");
      std::cout << i.second.get() << std::endl;
  }

  estimates_.at(choice)->update(value);
}

} // namespace bandits
