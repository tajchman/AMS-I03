project(Ex_6_1 LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

option(ENABLE_CUDA "Enable Cuda" OFF)
option(ENABLE_OPENCL "Enable OpenCL" OFF)

find_package(PNG REQUIRED)
include_directories(
  ${PNG_PNG_INCLUDE_DIR}
  util
)
add_compile_definitions(INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

add_subdirectory(util)
add_subdirectory(cpu)
add_subdirectory(gpu)

