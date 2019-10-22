#include <torch/torch.h>

#include <cstdint>
#include <chrono>
#include <iostream>

typedef std::chrono::nanoseconds ns;
constexpr uint64_t iters = 100000;

int main() {

  const auto start = std::chrono::high_resolution_clock::now();

  for (auto i = decltype(iters){0}; i < iters; ++i) {
    int a = 3;
    int b = 4;
    auto c = a + b;
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed = end - start;
  const auto elapsed_ns = std::chrono::duration_cast<ns>(elapsed);
  std::cout << "Ran in " << elapsed_ns.count() << "ns\n" << std::endl;
  std::cout << "Each iteration took ~" << elapsed_ns.count()/iters << "ns\n" << std::endl;
}
