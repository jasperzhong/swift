#include <torch/torch.h>

#include <ATen/ATen.h>
#include <ATen/core/functional.h> // fmap
#include <torch/csrc/jit/fuser/tensor_desc.h>

#include <cstdint>
#include <chrono>
#include <iostream>

typedef std::chrono::nanoseconds ns;
constexpr uint64_t iters = 100000;

using torch::jit::fuser::TensorDesc;

int main() {

  auto a = torch::ones({2, 3, 4});
  auto b = torch::ones({3, 4});
  std::vector<torch::Tensor> inputs{a, b};
  std::vector<TensorDesc> descs = c10::fmap<TensorDesc>(inputs);

  const auto start = std::chrono::high_resolution_clock::now();

  for (auto i = decltype(iters){0}; i < iters; ++i) {
    const size_t hash_code = torch::get_hash(0, inputs.size(), inputs);
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed = end - start;
  const auto elapsed_ns = std::chrono::duration_cast<ns>(elapsed);
  std::cout << "Ran in " << elapsed_ns.count() << "ns\n" << std::endl;
  std::cout << "Each iteration took ~" << elapsed_ns.count()/iters << "ns\n" << std::endl;
}
