#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <torch/torch.h>

const std::array<autotune::api::Implementation, 3> choices{
    autotune::api::Implementation::kConv2D_Native,
    autotune::api::Implementation::kConv2D_NNPack,
    autotune::api::Implementation::kConv2D_MKL};

std::string time_convolution(
    int64_t c_in,
    int64_t c_out,
    int64_t image_h,
    int64_t image_w,
    size_t choice) {
  int64_t batch_size = 1;
  int64_t kernel_size = 1;
  auto x = at::ones({batch_size, c_in, image_h, image_w});
  auto weight = at::ones({c_out, c_in, kernel_size, kernel_size});

  auto start = std::chrono::high_resolution_clock::now();
  autotune::convolution_2D(x, weight, choices[choice]);
  auto end = std::chrono::high_resolution_clock::now();
  auto delta_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return autotune::utils::string_format(
      "{\"choice\": %d, \"c_in\": %4d, \"c_out\": %4d, "
      "\"image_h\": %4d, \"image_w\": %4d, \"delta_ns\":  %11d}",
      choice,
      c_in,
      c_out,
      image_h,
      image_w,
      delta_ns);
}

void benchmark_grid(
    int64_t image_h,
    int64_t image_w,
    size_t seed,
    std::string log_file) {

  std::mt19937 engine(seed);
  std::uniform_int_distribution<size_t> impl_distribution(
      0, choices.size() - 1);
  std::uniform_int_distribution<int64_t> c_distribution(1, 96);

  size_t n = 5000;
  for (size_t i = 0; i < n; i++) {
    auto choice = impl_distribution(engine);
    auto c_in = c_distribution(engine);
    auto c_out = c_distribution(engine);
    auto result =
        time_convolution(c_in, c_out, image_h, image_w, choice);
    autotune::api::log(result);
  }
  autotune::api::flush_logs(log_file);
}

void gridsearch(int argc, char* argv[]) {
  std::string log_file = "/tmp/gridsearch_results_batch4.txt";
  std::remove(log_file.c_str());
  torch::set_num_threads(1);
  autotune::api::enable_logging();
  for (size_t i = 0; i < 50; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    benchmark_grid(128, 128, /*seed=*/i, log_file);
    auto end = std::chrono::high_resolution_clock::now();
    auto t =
        std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << "Batch: " << i << " done. " << t << "sec" << std::endl;
  }
}

void drill_down(int argc, char* argv[]) {
  std::string log_file = "/tmp/drilldown_results.txt";
  std::remove(log_file.c_str());
  torch::set_num_threads(1);
  autotune::api::enable_logging();

  int64_t image_h = 128;
  int64_t image_w = 128;

  // c_in, c_out
  std::vector<std::vector<int64_t>> configurations{
      {32, 32}, {32, 36}, {36, 32}, {36, 36}};

  std::mt19937 engine(0);
  std::uniform_int_distribution<size_t> impl_distribution(
      0, choices.size() - 1);
  std::uniform_int_distribution<size_t> cfg_distribution(
      0, configurations.size() - 1);

  size_t n = 10000;
  for (size_t i = 0; i < n; i++) {
    auto choice = impl_distribution(engine);
    auto cfg_choice = configurations[cfg_distribution(engine)];
    auto c_in = cfg_choice[0];
    auto c_out = cfg_choice[1];
    auto result = time_convolution(c_in, c_out, image_h, image_w, choice);
    autotune::api::log(result);
  }
  autotune::api::flush_logs(log_file);
}

const std::map<std::string, std::function<void(int, char**)>> tasks{
    {"gridsearch", &gridsearch},
    {"drilldown", &drill_down},
};

int main(int argc, char* argv[]) {
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
