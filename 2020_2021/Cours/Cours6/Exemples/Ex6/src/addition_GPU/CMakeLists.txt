cmake_minimum_required(VERSION 3.18)

get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${CODE} C CXX CUDA)

find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(
  util
)

add_executable(${CODE}
  main.cxx
  calcul_GPU.cu)


install(TARGETS ${CODE} DESTINATION .)
