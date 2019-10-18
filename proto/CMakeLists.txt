cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(hello_world)

find_package(Torch REQUIRED)

add_executable(hello_world hello_world.cpp)
target_link_libraries(hello_world "${TORCH_LIBRARIES}")
set_property(TARGET hello_world PROPERTY CXX_STANDARD 11)



