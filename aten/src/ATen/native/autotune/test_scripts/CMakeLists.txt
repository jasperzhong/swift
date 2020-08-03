cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(benchmark_convolution)

find_package(Torch REQUIRED)

add_executable(benchmark_convolution benchmark_convolution.cpp)
target_link_libraries(benchmark_convolution "${TORCH_LIBRARIES}")
set_property(TARGET benchmark_convolution PROPERTY CXX_STANDARD 14)
