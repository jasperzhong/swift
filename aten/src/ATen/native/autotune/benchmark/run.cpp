#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

const std::array<autotune::api::Implementation, 3> choices{
    autotune::api::Implementation::kConv2D_Native,
    autotune::api::Implementation::kConv2D_NNPack,
    autotune::api::Implementation::kConv2D_MKL};

const std::map<std::string, autotune::api::Implementation> str_to_choice{
  {"Native", autotune::api::Implementation::kConv2D_Native},
  {"NNPack", autotune::api::Implementation::kConv2D_NNPack},
  {"MKL", autotune::api::Implementation::kConv2D_MKL}
};

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
    auto result = time_convolution(c_in, c_out, image_h, image_w, choice);
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




std::vector<std::vector<int64_t>> product(std::vector<std::vector<int64_t>> x) {
    int64_t n = 1;
    for (auto xi : x) {
        n *= xi.size();
    }

    std::vector<std::vector<int64_t>> out;
    out.reserve(n);
    for (int64_t i = 0; i < n; i++) {
        out.push_back({});
    }

    int64_t stride = 1;
    for (auto xi : x) {
        int64_t index = 0;
        auto xi_size = xi.size();
        for (int64_t i = 0; i < n; i++) {
            out[i].push_back(xi[index]);
            if (!((i + 1) % stride))
                index = (index + 1) % xi_size;
        }
        stride *= xi_size;
    }

    return out;
}

void contextual(int argc, char* argv[]) {
  std::string log_file = "/tmp/contextual_results.txt";
  std::remove(log_file.c_str());

  torch::set_num_threads(1);
  autotune::api::set_active_bandit(
      autotune::api::AvailableBandits::kRandomChoice);

  { // Warmup
    auto x = at::ones({1, 1, 16, 16});
    auto weight = at::ones({1, 1, 1, 1});
    for (size_t i = 0; i < 100; i++) {
      autotune::convolution_2D(x, weight);
    }
  }

  auto product_cfg = product({
        // Batch size
        {1, 2, 4, 8, 16},

        // c_in
        {1, 4, 16, 32, 64, 128},

        // c_out
        {1, 4, 16, 32, 64, 128},

        // kernel_h
        {1, 3, 5, 7},

        // kernel_w
        {1, 3, 5, 7},

        // image_h
        {4, 16, 64, 128, 256},

        // image_w
        {4, 16, 64, 128, 256}
    });

    std::vector<std::array<std::array<int64_t, 4>, 2>> configurations;
    for (auto cfg_i : product_cfg) {
      auto batch_size = cfg_i[0];
      auto c_in = cfg_i[1];
      auto c_out = cfg_i[2];
      auto kernel_h = cfg_i[3];
      auto kernel_w = cfg_i[4];
      auto image_h = cfg_i[5];
      auto image_w = cfg_i[6];

      auto approx_work = batch_size * c_in * c_out * kernel_h * kernel_w * image_h * image_w;
      if (image_h < kernel_h || image_w < kernel_w || approx_work > 5e9)
        continue;

      std::array<int64_t, 4> x_size {batch_size, c_in, image_h, image_w};
      std::array<int64_t, 4> w_size {c_out, c_in, kernel_h, kernel_w};
      configurations.push_back({x_size, w_size});
    }

    std::cout << configurations.size() << " " << product_cfg.size() << std::endl;

  struct Distribution {
    std::vector<int64_t> choices;
    int64_t operator()(std::mt19937& engine) {
      std::uniform_int_distribution<int64_t> distribution(0, choices.size() - 1);
      return choices[distribution(engine)];
    }
  };

  std::mt19937 engine(0);
  // std::uniform_int_distribution<int64_t> channel_sizes(1, 128);
  // Distribution batch_sizes {{1, 2, 3, 8, 16}};
  // Distribution kernel_sizes {{1, 3, 5, 7}};
  // Distribution image_sizes {{32, 50, 64, 96, 128, 224, 256}};
  // auto coin_flip = [&](){ return std::uniform_int_distribution<>(0, 1)(engine); };

  // constexpr size_t n_candidates = 50000;
  // constexpr size_t n_conf_target = 1000;
  // constexpr int64_t max_approx_work = 5e8;
  // auto element_range = max_approx_work / (int64_t)n_conf_target;
  // std::array<bool, n_conf_target> conf_filled;
  // conf_filled.fill(false);
  // std::vector<std::array<std::array<int64_t, 4>, 2>> configurations;

  // for (size_t i = 0; i <  n_candidates; i++) {
  //   auto c_in = channel_sizes(engine);
  //   auto c_out = channel_sizes(engine);
  //   auto batch_size = batch_sizes(engine);
  //   auto kernel_h = kernel_sizes(engine);
  //   auto kernel_w = coin_flip() ? kernel_h : kernel_sizes(engine);
  //   auto image_h = image_sizes(engine);
  //   auto image_w = coin_flip() ? image_h : image_sizes(engine);

  //   auto approx_work = batch_size * c_in * c_out * kernel_h * kernel_w * image_h * image_w;
  //   if (approx_work > max_approx_work)
  //     continue;

  //   auto index = approx_work / element_range;
  //   if (!conf_filled[index]) {
  //     conf_filled[index] = true;
  //     std::array<int64_t, 4> x_size {batch_size, c_in, image_h, image_w};
  //     std::array<int64_t, 4> w_size {c_out, c_in, kernel_h, kernel_w};
  //     configurations.push_back({x_size, w_size});

  //     if (configurations.size() == n_conf_target)
  //       break;
  //   }
  // }

  // std::cout << configurations.size() << std::endl;



  Distribution num_threads {{1, 2, 4, 8, 16}};
  std::uniform_int_distribution<int64_t> conf_distribution(0, configurations.size() - 1);
  autotune::api::enable_logging();
  for (size_t i = 0; i < 10000000; i++){
    if (!(i % 1000)) {
      // Set threads in blocks since it mucks with the underlying runtime.
      torch::set_num_threads(num_threads(engine));
    }
    auto cfg_choice = conf_distribution(engine);
    auto x = at::ones(configurations[cfg_choice][0]);
    auto weight = at::ones(configurations[cfg_choice][1]);
    autotune::convolution_2D(x, weight);

    if (!((i + 1) % 1000)) {
      std::cout << i + 1 << " complete." << std::endl;
      autotune::api::flush_logs(log_file);
    }
  }

  autotune::api::flush_logs(log_file);
}

void profile(int argc, char* argv[]) {
  int64_t batch_size, c_in, c_out, image_h, image_w;
  torch::set_num_threads(1);
  if (argc == 2) {
    std::string arg1 = *(argv + 1);
    if (arg1 == "1,1024,1024,4,4") {
      batch_size = 1;
      c_in = 1024;
      c_out = 1024;
      image_h = 4;
      image_w = 4;
    } else {
      std::cout << "Unknown: " << arg1 << std::endl;
      return;
    }
  } else if (argc == 3){
    std::string arg1 = *(argv + 1);
    std::string arg2 = *(argv + 2);

    batch_size = 1;
    c_in = std::stoi(arg1);
    c_out = std::stoi(arg2);
    image_h = 128;
    image_w = 128;
  } else {
    std::cout << "Expected 3 args, got " << argc << std::endl;
    return;
  }

  std::string arg0 = *argv;

  // auto x = at::ones({1, c_in, 256, 256});
  auto x = at::ones({batch_size, c_in, image_h, image_w});
  auto weight = at::ones({c_out, c_in, 3, 3});

  auto start = std::chrono::high_resolution_clock::now();
  auto elapsed = [start](){
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  };

  size_t n = 0;
  while (elapsed() < 5e9){
    autotune::convolution_2D(x, weight, str_to_choice.at(arg0));
    n++;
  }

  auto delta_ns = elapsed();
  std::cout << (double)(delta_ns / n) / 1e6 << " ms" << std::endl;
}

const std::map<std::string, std::function<void(int, char**)>> tasks{
    {"gridsearch", &gridsearch},
    {"drilldown", &drill_down},
    {"contextual", &contextual},
    {"profile", &profile},
};

int main(int argc, char* argv[]) {
  if (argc == 1 || tasks.find(*(argv + 1)) == tasks.end()) {
    std::cout << "Tasks: " << std::endl;
    for (auto t : tasks)
      std::cout << "  " << t.first << std::endl;
    return 0;
  }

  auto task = *(argv + 1);
  tasks.at(task)(argc - 2, argv + 2);
  return 0;
}
