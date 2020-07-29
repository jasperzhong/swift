cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

find_package(Torch REQUIRED)

add_executable(test test.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 14)
