#include <ATen/native/autotune/dispatch/common.h>

#include <forward_list>
#include <functional>
#include <limits>

#include <c10/util/Exception.h>

namespace autotune {
namespace selection {

bool KernelEntryPoint::fallback() {
  return false;
}

KernelEntryPoint::supported_implementations KernelEntryPoint::implementations() {
    KernelEntryPoint::supported_implementations output;
    for (auto c : costs()) {
        output.push_back(c.impl);
    }
    return output;
}

KernelEntryPoint::map_key KernelEntryPoint::key() {
  return {
      task(),
      static_cast<uint32_t>(hash_x0_ % std::numeric_limits<uint32_t>::max()),
      hash_x1_};
}

void KernelEntryPoint::compute_hash(
    std::forward_list<c10::IntArrayRef> features) {
  hash_x0_ = 0;
  hash_x1_ = 1;
  for (auto f : features) {
    hash(hash_x0_, f);
    hash(hash_x1_, f);
  }

  hash_computed_ = true;
}

// https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
void KernelEntryPoint::hash(uint64_t& x, c10::IntArrayRef features) {
  for (auto f : features) {
    x ^= std::hash<int64_t>{}(f) + 0x9e3779b9 + (x << 6) + (x >> 2);
  }
}

} // namespace selection
} // namespace autotune
