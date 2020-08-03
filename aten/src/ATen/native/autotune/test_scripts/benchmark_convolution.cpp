#include <functional>
#include <iostream>
#include <map>
#include <string>

#include <torch/torch.h>


void debug(int argc, char* argv[]) {
  torch::set_num_threads(1);
  autotune::api::enable_logging();
  autotune::api::set_active_bandit(
    autotune::api::AvailableBandits::kRandomChoice);
  {
    auto x = at::ones({1, 36, 128, 128});
    auto weight = at::ones({32, 36, 1, 1});
    for (size_t i = 0; i < 200; i++) {
      autotune::convolution_2D(x, weight);
    }
  }

  autotune::api::log("");
  autotune::api::set_active_bandit(
      autotune::api::AvailableBandits::kGaussian);

//   {
//     auto x = at::ones({1, 36, 128, 128});
//     auto weight = at::ones({32, 36, 1, 1});
//     for (size_t i = 0; i < 40; i++) {
//       autotune::convolution_2D(x, weight);
//     }
//   }

//   autotune::api::log("");
//   {
//     auto x = at::ones({1, 36, 128, 128});
//     auto weight = at::ones({32, 36, 1, 1});
//     for (size_t i = 0; i < 40; i++) {
//       auto evict = at::ones({16, 32, 128, 128});
//       autotune::convolution_2D(x, weight);
//     }
//   }

  autotune::api::log("");
  {
    auto x = at::ones({1, 32, 128, 128});
    auto weight = at::ones({32, 32, 1, 1});
    for (size_t i = 0; i < 500; i++) {
    //   auto evict = at::ones({16, 32, 128, 128});
      autotune::convolution_2D(x, weight);
    }
  }

  autotune::api::flush_logs(std::cout);
}

const std::map<std::string, std::function<void(int, char**)>> tasks {
    {"debug", &debug}
};

int main (int argc, char *argv[]) {
    if (argc == 1 || tasks.find(*(argv + 1)) == tasks.end()) {
        std::cout << "Tasks: " << std::endl;
        for (auto t : tasks)
            std::cout << "  " << t.first << std::endl;
        return 0;
    }

    auto task = *(argv + 1);
    std::cout << "Task: " << task << std::endl;
    tasks.at(task)(argc - 2, argv + 2);
    return 0;
}
