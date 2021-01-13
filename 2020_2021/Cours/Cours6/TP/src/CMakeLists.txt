project(matrices CUDA C CXX)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CUDA_ARCHITECTURES "off")

add_subdirectory(cpu)
add_subdirectory(gpu)

