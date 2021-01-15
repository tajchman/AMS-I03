project(EX_6_2 CUDA C CXX)

cmake_minimum_required(VERSION 2.8)

find_package(PNG)
include_directories(
  ${PNG_PNG_INCLUDE_DIR}
  util
)

add_compile_definitions(INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
 
add_subdirectory(cpu)
add_subdirectory(gpu)

