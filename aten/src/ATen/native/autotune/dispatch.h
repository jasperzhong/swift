#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>

#include <ATen/native/autotune/bandit.h>
#include <ATen/native/autotune/definitions.h>
#include <ATen/native/autotune/estimator.h>

namespace autotune {

class CostDispatcher {
 public:
  static CostDispatcher& singleton() {
    static CostDispatcher _singleton;
    return _singleton;
  }

  using bandit_ptr = std::shared_ptr<bandits::GaussianBandit>;
  class RAII_Choice {
   public:
    RAII_Choice(const RAII_Choice&) = delete;
    RAII_Choice(RAII_Choice&&) = default;
    RAII_Choice(DispatchChoice choice, bandit_ptr b);
    // ~RAII_Choice();
    DispatchChoice get() {
      return choice_;
    }
    void finished();

   private:
    DispatchChoice choice_;
    bandit_ptr b_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  };
  RAII_Choice choose(EntryPoint e);
  std::shared_ptr<ImplStats> get_impl_stats(autotune::DispatchChoice choice);

 private:
  CostDispatcher() {
    for (size_t i = 0; i < NumDispatchChoice; i++) {
      impl_stats_[i] = std::make_shared<ImplStats>();
    }
  };

  std::array<std::shared_ptr<ImplStats>, NumDispatchChoice> impl_stats_;
  std::map<EntryPoint::map_key, bandit_ptr> bandits_;
  uint64_t next_seed_ = 0;
};

} // namespace autotune
