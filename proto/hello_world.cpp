#include <torch/torch.h>

#include <iostream>

int main() {
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;

  const auto version = torch::proto::version();
  std::cout << version << std::endl;
}