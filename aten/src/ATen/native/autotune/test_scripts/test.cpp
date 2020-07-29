#include <iostream>
#include <vector>

#include <torch/torch.h>


int main() {
  torch::set_num_threads(1);

  // Warmup of sorts.
  autotune::set_drunken_bandit();
  autotune::test_bandit(1000);

  autotune::set_gaussian_bandit();
  autotune::test_bandit(15000);

  autotune::set_drunken_bandit();
  autotune::test_bandit(15000);

  autotune::flush_results();
}
