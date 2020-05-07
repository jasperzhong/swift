#pragma once

#include <cstdint>
#include <string>

#include <ATen/ATen.h>

// Scatter / Gather heuristics
namespace sg_heuristic {

CAFFE2_API void set(std::string method_name, std::string value);

enum class Method {
  GATHER,
  SCATTER,
  SCATTER_FILL,
  SCATTER_ADD,
  INDEX_SELECT,
};

enum class LoopSpecialization {
  AUTO,
  ONE_DIMENSIONAL,
  ONE_DIMENSIONAL_CONTIGUOUS,
  BATCH_MAJOR,
  BATCH_MAJOR_CONTIGUOUS,
  FEATURE_MAJOR,
};

LoopSpecialization choose_specialization(
    Method method,
    const bool vector_subtask,
    const bool contiguous_subtask,
    const int64_t n,
    const int64_t self_dim_stride,
    const int64_t index_dim_size);

} // namespace sg_heuristic
