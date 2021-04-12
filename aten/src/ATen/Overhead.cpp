#include <ATen/ATen.h>

namespace at {
namespace native {

Tensor _noop_unary_cpu(const Tensor& self) {
  return self;
}

Tensor _noop_unary_cuda(const Tensor& self) {
  return self;
}

Tensor _noop_binary_cpu(const Tensor& self, const Tensor& other) {
  return self;
}

Tensor _noop_binary_cuda(const Tensor& self, const Tensor& other) {
  return self;
}

}  // namespace native
}  // namespace at
