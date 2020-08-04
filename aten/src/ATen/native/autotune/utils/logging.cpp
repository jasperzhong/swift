#include <ATen/native/autotune/utils/logging.h>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <ATen/native/autotune/api.h>
#include <ATen/native/autotune/dispatch/common.h>
#include <ATen/native/autotune/utils/common.h>
#include <c10/util/flat_hash_map.h>

namespace autotune {
namespace logging {

std::map<api::AvailableBandits, std::string> bandit_str = {
    {api::AvailableBandits::kRandomChoice, "DrunkenBandit"},
    {api::AvailableBandits::kGaussian, "GaussianBandit"},
    {api::AvailableBandits::kNone, "None"}};

std::map<api::Implementation, std::string> impl_str = {
    {api::Implementation::kConv2D_Native, "Conv2D_Native"},
    {api::Implementation::kConv2D_NNPack, "Conv2D_NNPack"},
    {api::Implementation::kConv2D_MKL, "Conv2D_MKL"},
    {api::Implementation::kDisabled, "Disabled"},
    {api::Implementation::kFallback, "Fallback"},
    {api::Implementation::kUnsupported, "Unsupported"},
};

ska::flat_hash_map<
    selection::KernelEntryPoint::MapKey,
    std::string,
    selection::KernelEntryPoint::Hash>
    key_reprs;

std::vector<std::string> records;
bool logging_enabled{false};

void register_key(
    selection::KernelEntryPoint::MapKey key,
    std::function<std::string()> repr) {
  if (!logging_enabled)
    return;
  if (key_reprs.find(key) == key_reprs.end()) {
    key_reprs[key] = repr();
    records.push_back(utils::string_format("Repr: %s", key_reprs[key].c_str()));
  }
}

std::string to_string(selection::KernelEntryPoint::MapKey key) {
  return key_reprs.at(key);
}

std::string to_string(api::AvailableBandits bandit) {
  return bandit_str.at(bandit);
}

std::string to_string(api::Implementation choice) {
  return impl_str.at(choice);
}

void record(
    api::AvailableBandits bandit,
    selection::KernelEntryPoint::MapKey key,
    api::Implementation choice,
    size_t delta_ns) {
  if (!logging_enabled)
    return;
  records.push_back(utils::string_format(
      "%s     %-70s     %-14s     %10d",
      to_string(bandit).c_str(),
      to_string(key).c_str(),
      to_string(choice).c_str(),
      delta_ns));
}

} // namespace logging

namespace api {
void enable_logging() {
  logging::logging_enabled = true;
}
void disable_logging() {
  logging::logging_enabled = false;
}

void log(std::string s) {
  if (!logging::logging_enabled)
    return;
  logging::records.push_back(s);
}

void flush_logs(std::string filename) {
  std::ofstream out;
  out.open(filename, std::ios_base::app);
  for (auto r : logging::records) {
    out << r << std::endl;
  }
  logging::records.clear();
  out.close();
}

void flush_logs(std::ostream& out) {
  for (auto r : logging::records) {
    out << r << std::endl;
  }
  logging::records.clear();
}
} // namespace api
} // namespace autotune
