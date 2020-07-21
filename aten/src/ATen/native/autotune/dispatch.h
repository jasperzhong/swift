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
  using correction_ptr = std::shared_ptr<autotune::GaussianEstimator>;
  class RAII_Choice {
   public:
    RAII_Choice(const RAII_Choice&) = delete;
    RAII_Choice(RAII_Choice&&) = default;
    RAII_Choice(EntryPoint::map_key key, DispatchChoice choice, bandit_ptr b);
    ~RAII_Choice();
    DispatchChoice get() {
      return choice_;
    }

   private:
    EntryPoint::map_key key_;
    DispatchChoice choice_;
    bandit_ptr b_;
    std::shared_ptr<size_t> impl_called_count_;

    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  };
  RAII_Choice choose(EntryPoint e);

 private:
  CostDispatcher() {
    for (size_t i = 0; i < prior_correction_.max_size(); i++) {
      // TODO: better heuristic for initial variance.
      prior_correction_[i] =
          std::make_shared<autotune::GaussianEstimator>(1.0, 1.0 / 9.0);
    }
  };
  std::map<EntryPoint::map_key, bandit_ptr> bandits_;
  std::array<correction_ptr, NumDispatchChoice> prior_correction_;
  uint64_t next_seed_ = 0;
};

} // namespace autotune
