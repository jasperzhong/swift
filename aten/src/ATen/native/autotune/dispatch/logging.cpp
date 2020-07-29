#include <ATen/native/autotune/dispatch/logging.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <ATen/native/autotune/dispatch/common.h>
#include <ATen/native/autotune/dispatch/core.h>
#include <ATen/native/autotune/kernels/common.h>

namespace autotune {
namespace logging {

using AvailableBandits = selection::DispatchInterface::AvailableBandits;
struct Record;

std::map<AvailableBandits, std::string> bandit_str = {
    {AvailableBandits::kRandomChoice, "DrunkenBandit"},
    {AvailableBandits::kGaussian, "GaussianBandit"},
    {AvailableBandits::kNone, "None"}};

std::map<kernels::Implementation, std::string> impl_str = {
  {kernels::Implementation::kConv2D_Native, "Conv2D_Native"},
  {kernels::Implementation::kConv2D_NNPack, "Conv2D_NNPack"},
  {kernels::Implementation::kConv2D_MKL, "Conv2D_MKL"},
  {kernels::Implementation::kDisabled, "Disabled"},
  {kernels::Implementation::kFallback, "Fallback"},
  {kernels::Implementation::kUnsupported, "Unsupported"},
};

std::map<selection::KernelEntryPoint::map_key, std::string> key_reprs;
std::vector<Record> records;

struct Record {
  AvailableBandits bandit;
  selection::KernelEntryPoint::map_key key;
  kernels::Implementation choice;
  size_t delta_ns;
};

void register_key(
    selection::KernelEntryPoint::map_key key,
    std::function<std::string()> repr) {
  if (key_reprs.find(key) == key_reprs.end()) {
    key_reprs[key] = repr();
    std::cout << "Repr: " << key_reprs[key] << std::endl;
  }
}

void record(
    AvailableBandits bandit,
    selection::KernelEntryPoint::map_key key,
    kernels::Implementation choice,
    size_t delta_ns) {
  records.push_back({bandit, key, choice, delta_ns});
}

void flush() {
  auto out_file = "/data/users/taylorrobie/repos/pytorch/aten/src/ATen/native/autotune/test_scripts/results.txt";
  std::remove(out_file);

  std::ofstream out(out_file);
  for (auto r : records) {
    out << bandit_str.at(r.bandit) << "      " << key_reprs.at(r.key) << "      "
        << impl_str.at(r.choice) << "      " << r.delta_ns << std::endl;
  }
  out.close();
}
} // namespace logging
} // namespace autotune
