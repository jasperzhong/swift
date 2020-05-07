#include <ATen/native/Heuristics.h>

#include <array>
#include <cstdint>
#include <map>
#include <string>

#include <ATen/ATen.h>

namespace sg_heuristic {

// Conversion tables for Python to C++
const std::map<std::string, Method> STR_TO_METHOD(
    {{"gather",       Method::GATHER},
     {"scatter",      Method::SCATTER},
     {"scatter_fill", Method::SCATTER_FILL},
     {"scatter_add",  Method::SCATTER_ADD},
     {"index_select", Method::INDEX_SELECT}}
);

const std::map<std::string, LoopSpecialization> STR_TO_SPECIALIZATION(
    {{"auto",          LoopSpecialization::AUTO},
     {"batch_major",   LoopSpecialization::BATCH_MAJOR},
     {"feature_major", LoopSpecialization::FEATURE_MAJOR}}
);

// Current heuristic value.
std::map<Method, LoopSpecialization> CURRENT_HEURISTIC({
    {Method::GATHER,       LoopSpecialization::AUTO},
    {Method::SCATTER,      LoopSpecialization::AUTO},
    {Method::SCATTER_FILL, LoopSpecialization::AUTO},
    {Method::SCATTER_ADD,  LoopSpecialization::AUTO},
    {Method::INDEX_SELECT, LoopSpecialization::AUTO}
});

CAFFE2_API void set(std::string method_name, std::string value){
    auto method = STR_TO_METHOD.at(method_name);
    auto specialization = STR_TO_SPECIALIZATION.at(value);
    CURRENT_HEURISTIC[method] = specialization;
}

LoopSpecialization choose_specialization(
    Method method,
    const bool vector_subtask,
    const bool contiguous_subtask,
    const int64_t n,
    const int64_t self_dim_stride,
    const int64_t index_dim_size) {
  auto specialization = CURRENT_HEURISTIC.at(method);
  if (specialization != LoopSpecialization::AUTO) {
    return specialization;
  };

  if (vector_subtask) {
    return contiguous_subtask ? LoopSpecialization::ONE_DIMENSIONAL_CONTIGUOUS
                              : LoopSpecialization::ONE_DIMENSIONAL;
  }

  if (contiguous_subtask) {
    return LoopSpecialization::BATCH_MAJOR_CONTIGUOUS;
  }

  return ((self_dim_stride == 1) || (n < index_dim_size))
      ? LoopSpecialization::FEATURE_MAJOR
      : LoopSpecialization::BATCH_MAJOR;
}

} // namespace sg_heuristic
