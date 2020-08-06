#pragma once

#include <forward_list>
#include <string>
#include <tuple>
#include <vector>

#include <ATen/native/autotune/api.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>

namespace autotune {
namespace selection {

constexpr unsigned approx_implementations_per_task = 3;

/*
Standardize kernel specific logic at the boundaries of dispatch.
  When a kernel is called, several determinations need to be made:
    1) Should autotuning be used at all, or should the system simply fall
       back to a particular static configuration?

    2) Which implementations are viable for a given task?

    3) What is the expected cost of each implementation?

  The answers to these questions can be represented in a standard form,
  however the signatures for the functions which make that determination
  vary by kernel. This class serves to standardize the inlet side of
  autotuning; each abstract task (e.g. convolution) creates a subclass of
  KernelEntryPoint which can take all relevant information as constructor
  arguments, and then the rest of the system operates on a single abstract
  interface.
*/
class KernelEntryPoint {
 public:
  struct CostEstimate {
    api::Implementation impl;
    double cost;
  };

  using cost_estimates =
      c10::SmallVector<CostEstimate, approx_implementations_per_task>;
  using supported_implementations =
      c10::SmallVector<api::Implementation, approx_implementations_per_task>;

  // Default behavior:
  //   fallback() returns false
  //   implementations() calls costs() and extracts implementations.
  virtual bool fallback();
  virtual api::Task task() = 0;
  virtual cost_estimates costs() = 0;
  virtual std::string repr() = 0;
  virtual supported_implementations implementations();

  // Subclass constructors are responsible for calling this function
  // with any features which affect the cost. (Unless they plan to return
  // true to `fallback()`, which allows them to skip hashing and save overhead
  // for cases where autotuning will not be used.)
  void declare_features(std::forward_list<c10::IntArrayRef> features);

  struct MapKey {
    std::vector<int64_t> data;
    bool operator==(const MapKey&) const;
  };
  const MapKey& key();

  struct Hash {
    size_t operator()(const KernelEntryPoint::MapKey&) const;
  };

 private:
  friend class Hash;
  MapKey key_;
};

} // namespace selection
} // namespace autotune
