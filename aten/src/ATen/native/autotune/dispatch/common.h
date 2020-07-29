#pragma once

#include <forward_list>
#include <string>
#include <tuple>

#include <ATen/native/autotune/kernels/common.h>
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
    kernels::Implementation impl;
    double cost;
  };

  using cost_estimates =
      c10::SmallVector<CostEstimate, approx_implementations_per_task>;
  using supported_implementations = c10::
      SmallVector<kernels::Implementation, approx_implementations_per_task>;
  using map_key = std::tuple<kernels::Task, uint32_t, uint64_t>;

  // Default behavior:
  //   fallback() returns false
  //   implementations() calls costs() and extracts implementations.
  virtual bool fallback();
  virtual kernels::Task task() = 0;
  virtual cost_estimates costs() = 0;
  virtual std::string repr() = 0;
  virtual supported_implementations implementations();

  // Subclass constructors are responsible for calling this function
  // with any features which affect the cost. (Unless they plan to return
  // true to `fallback()`, which allows them to skip hashing and save overhead
  // for cases where autotuning will not be used.)
  void compute_hash(std::forward_list<c10::IntArrayRef> features);
  map_key key();

 private:
  void hash(uint64_t& x, c10::IntArrayRef features);
  bool hash_computed_{false};
  uint64_t hash_x0_;
  uint64_t hash_x1_;
};

static_assert(sizeof(KernelEntryPoint::map_key) == 16);

} // namespace selection
} // namespace autotune
