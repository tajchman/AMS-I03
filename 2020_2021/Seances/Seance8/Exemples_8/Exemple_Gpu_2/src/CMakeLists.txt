project(EX_6_2 C CXX)

cmake_minimum_required(VERSION 3.1)

set(BUILD_SHARE_LIBS false)

option(ENABLE_CUDA "Enable Cuda" OFF)
option(ENABLE_OPENCL "Enable OpenCL" OFF)

find_package(PNG)
include_directories(
  ${PNG_PNG_INCLUDE_DIR}
  util
)
message(PNG_LIBRARIES="${PNG_LIBRARIES}")

add_compile_definitions(INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
 
add_subdirectory(cpu)
add_subdirectory(gpu)

