cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(proto_tensor)

find_package(Torch REQUIRED)

add_executable(hello_world hello_world.cpp)
target_link_libraries(hello_world "${TORCH_LIBRARIES}")
set_property(TARGET hello_world PROPERTY CXX_STANDARD 11)

add_executable(scalar_addition scalar_addition.cpp)
target_link_libraries(scalar_addition "${TORCH_LIBRARIES}")
set_property(TARGET scalar_addition PROPERTY CXX_STANDARD 11)

add_executable(array_addition array_addition.cpp)
target_link_libraries(array_addition "${TORCH_LIBRARIES}")
set_property(TARGET array_addition PROPERTY CXX_STANDARD 11)

add_executable(tensor_addition tensor_addition.cpp)
target_link_libraries(tensor_addition "${TORCH_LIBRARIES}")
set_property(TARGET tensor_addition PROPERTY CXX_STANDARD 11)

add_executable(hash_benchmark hash_benchmark.cpp)
target_link_libraries(hash_benchmark "${TORCH_LIBRARIES}")
set_property(TARGET hash_benchmark PROPERTY CXX_STANDARD 11)

add_executable(tensor_addition tensor_addition.cpp)
target_link_libraries(tensor_addition "${TORCH_LIBRARIES}")
set_property(TARGET tensor_addition PROPERTY CXX_STANDARD 11)
