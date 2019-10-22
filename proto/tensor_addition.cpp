#include <torch/torch.h>

#include <cstdint>
#include <chrono>
#include <iostream>

typedef std::chrono::nanoseconds ns;
constexpr uint64_t iters = 100000;

int main() {

  const auto start = std::chrono::high_resolution_clock::now();

  // for (auto i = decltype(iters){0}; i < iters; ++i) {
  //   std::array<int, 3> a{1, 2, 3};
  //   std::array<int, 3> b{4, 5, 6};
  //   std::array<int, 3> c{0, 0, 0};
  //   for (auto i = decltype(c.size()){0}; i < c.size(); ++i) {
  //     c[i] = a[i] + b[i];
  //   }
  // }

  const auto echo = torch::proto::echo();
  std::cout << echo << std::endl;

  const auto tensor_echo = torch::proto::tensor_echo();
  std::cout << tensor_echo << std::endl;

  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed = end - start;
  const auto elapsed_ns = std::chrono::duration_cast<ns>(elapsed);
  std::cout << "Ran in " << elapsed_ns.count() << "ns\n" << std::endl;
  std::cout << "Each iteration took ~" << elapsed_ns.count()/iters << "ns\n" << std::endl;
}
