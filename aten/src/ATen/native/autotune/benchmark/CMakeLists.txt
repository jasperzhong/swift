cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(benchmark)

find_package(Torch REQUIRED)

add_executable(benchmark_autotune run.cpp)
target_link_libraries(benchmark_autotune "${TORCH_LIBRARIES}")

set_property(TARGET benchmark_autotune PROPERTY CXX_STANDARD 14)
