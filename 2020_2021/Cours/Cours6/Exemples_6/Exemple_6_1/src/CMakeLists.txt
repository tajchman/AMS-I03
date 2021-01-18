project(Ex_6_1 C CXX)
cmake_minimum_required(VERSION 3.1)

option(ENABLE_CUDA "Enable Cuda" OFF)
if (ENABLE_CUDA)
  enable_language(CUDA)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(util)

foreach(d
    addition_CPU
    addition_CPU_OpenMP
    addition_GPU
    addition_GPU_reduction
    Pi_CPU
    Pi_GPU)
   add_subdirectory(${d})
endforeach()


