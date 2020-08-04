#include <ATen/native/autotune/dispatch/common.h>

#include <forward_list>
#include <functional>
#include <limits>

#include <ATen/native/autotune/api.h>
#include <c10/util/Exception.h>

namespace autotune {
namespace selection {

bool KernelEntryPoint::fallback() {
  return false;
}

KernelEntryPoint::supported_implementations KernelEntryPoint::
    implementations() {
  KernelEntryPoint::supported_implementations output;
  for (auto c : costs()) {
    output.push_back(c.impl);
  }
  return output;
}

void KernelEntryPoint::declare_features(
    std::forward_list<c10::IntArrayRef> features) {
  key_.data.clear();

  auto t = task();
  auto impls = implementations();
  size_t feature_size = 1 + impls.size();
  for (auto f : features) {
    feature_size += 1 + f.size();
  }

  key_.data.reserve(feature_size);
  key_.data.push_back(static_cast<int64_t>(t));
  for (auto impl : impls) {
    key_.data.push_back(static_cast<int64_t>(impl));
  }

  for (auto f : features) {
    // int64_t min serves to demarkate the otherwise flat vector.
    key_.data.push_back(std::numeric_limits<int64_t>::min());
    for (auto fi : f) {
      key_.data.push_back(fi);
    }
  }
}

bool KernelEntryPoint::MapKey::operator==(const KernelEntryPoint::MapKey& other) const {
  TORCH_INTERNAL_ASSERT(data.size() && other.data.size());
  return (
      data.size() == other.data.size() &&
      std::equal(data.begin(), data.end(), other.data.begin()));
}

const KernelEntryPoint::MapKey& KernelEntryPoint::key() {
  return key_;
}

// https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
size_t KernelEntryPoint::Hash::operator()(const KernelEntryPoint::MapKey& key) const {
  TORCH_INTERNAL_ASSERT(key.data.size());
  auto x = std::hash<int64_t>{}(0);
  for (auto i : key.data) {
    x ^= std::hash<int64_t>{}(i) + 0x9e3779b9 + (x << 6) + (x >> 2);
  }
  return x;
}

} // namespace selection
} // namespace autotune

namespace at {
namespace native {
bool set_autotune(std::string command) {
  using namespace autotune::api;
  if (command == "on") {
    at::_nnpack_available(); // Init NNPack
    set_active_bandit(AvailableBandits::kGaussian);
  } else if (command == "random on") {
    at::_nnpack_available(); // Init NNPack
    set_active_bandit(AvailableBandits::kRandomChoice);
  } else if (command == "logging on") {
    enable_logging();
  } else if (command == "off") {
    set_active_bandit(AvailableBandits::kNone);
    disable_logging();
  } else if (command == "flush") {
    flush_logs(std::cout);
  } else if (command == "summarize") {
    summarize();
  }
  return true;
}
}
}
