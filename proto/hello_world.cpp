#include <torch/torch.h>

#include <iostream>

int main() {
  std::cout << "Proto tensor benchmark example." << std::endl;

  std::cout << "Validating standard libtorch  with eye(3):" << std::endl;
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;

  const auto version = torch::proto::version();
  std::cout << "Current proto tensor version is: " << version << std::endl;
}
