#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <torch/torch.h>

static int64_t batch_size = 1;
static int64_t kernel_size = 1;
static std::vector<int64_t> stride{1, 1};
static std::vector<int64_t> dilation{1, 1};
static std::vector<int64_t> padding{0, 0};
static std::vector<int64_t> output_padding{0, 0};

static std::string log_file = "/tmp/gridsearch_results.txt";

enum class Impl {
  kTH = 0,
  kNNPack,
  kMKL,
};

struct Measurement {
    Impl choice;
    int64_t c_in;
    int64_t c_out;
    int64_t image_h;
    int64_t image_w;
    size_t delta_ns;
};

size_t time_convolution(int64_t c_in, int64_t c_out, int64_t image_h, int64_t image_w, Impl choice) {
  auto x = at::ones({batch_size, c_in, image_h, image_w});
  auto weight = at::ones({c_out, c_in, kernel_size, kernel_size});
  auto bias = at::ones({c_out});

  auto start = std::chrono::high_resolution_clock::now();
  switch (choice) {
    case Impl::kTH:
      at::thnn_conv2d(
          x, weight, {kernel_size, kernel_size}, bias, stride, padding);
      break;
    case Impl::kNNPack:
      at::_nnpack_spatial_convolution(x, weight, bias, padding, stride);
      break;
    case Impl::kMKL:
      at::mkldnn_convolution(
          x,
          weight,
          bias,
          padding,
          stride,
          dilation,
          /*groups=*/1);
      break;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto delta_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return delta_ns;
}

void flush(std::vector<Measurement>& measurements) {
    std::ofstream out;
    out.open(log_file, std::ios_base::app);
    for (auto m : measurements) {
        out << static_cast<size_t>(m.choice) << " " << m.c_in << " "
            << m.c_out << " " << m.image_h << "  " << m.image_w << " "
            << m.delta_ns << "\n";
    }
    out.flush();
    out.close();
    measurements.clear();
}

void benchmark_grid(int64_t image_h, int64_t image_w) {
  std::mt19937 engine(0);
  std::uniform_int_distribution<size_t> impl_distribution(0, 2);
  std::uniform_int_distribution<int64_t> c_distribution(1, 96);

  size_t n = 100000;
  size_t flush_cadence = 25000;
  std::vector<Measurement> measurements;
  measurements.reserve(flush_cadence);
  size_t running_count = 0;

  for (size_t i = 0; i < n; i++) {
    auto choice = static_cast<Impl>(impl_distribution(engine));
    auto c_in = c_distribution(engine);
    auto c_out = c_distribution(engine);
    auto delta_ns = time_convolution(c_in, c_out, image_h, image_w, choice);
    Measurement m {choice, c_in, c_out, image_h, image_w, delta_ns};
    measurements.push_back(m);

    if (measurements.size() == flush_cadence) {
        flush(measurements);
        running_count += flush_cadence;
        std::cout << running_count << " / " << n << std::endl;
    }
  }
  flush(measurements);
}

int main() {
  torch::set_num_threads(1);
  at::_nnpack_available(); // Init NNPack
  std::remove(log_file.c_str());

  for (size_t i = 0; i < 100; i++) {
    benchmark_grid(128, 128);
    std::cout << "Batch " << i << " finished." << std::endl;
  }

  std::cout << "done" << std::endl;

  return 0;
}
