#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include <ATen/ATen.h>
#include <c10/macros/Export.h>


namespace autotune {
namespace api {

// ============================================================================
// == Symbols =================================================================
// ============================================================================
enum class CAFFE2_API Task;
enum class CAFFE2_API Implementation;
enum class CAFFE2_API AvailableBandits;
bool enabled();
void CAFFE2_API set_active_bandit(AvailableBandits);
void CAFFE2_API enable_logging();
void CAFFE2_API disable_logging();
void CAFFE2_API log(std::string);
void CAFFE2_API flush_logs(std::string filename);
void CAFFE2_API flush_logs(std::ostream& out);
void CAFFE2_API summarize();
void CAFFE2_API reset();


// ============================================================================
// == Values ==================================================================
// ============================================================================
enum class Task {
  kConv2D = 0,

  kNotApplicable,
};

enum class Implementation {
  kConv2D_Native = 0,
  kConv2D_NNPack,
  kConv2D_MKL,

  // Autotuning has not been enabled in at::globalContext.
  kDisabled,

  // The kernel entry point has elected not to use autotuning.
  kFallback,

  // An implementation cannot be selected for an unknown reason.
  kUnsupported,

  TOTAL_COUNT, // Must be the final element.
};
constexpr size_t NumImplementations = (size_t)Implementation::TOTAL_COUNT;

enum class AvailableBandits {
  kRandomChoice,
  kGaussian,

  kNone,
};
} // namespace api

// Temporary. Evenentually this should be rolled into at::_convolution.
at::Tensor CAFFE2_API convolution_2D(at::Tensor& x, at::Tensor& weight);
at::Tensor CAFFE2_API convolution_2D(at::Tensor& x, at::Tensor& weight, api::Implementation);

namespace utils {
// https://stackoverflow.com/a/26221725
template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
  size_t size =
      snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
  if (size <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(
      buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}
} // namespace utils
} // namespace autotune
