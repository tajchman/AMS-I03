project(Ex_6_1 C CXX)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(util)

foreach(d
    addition_CPU
    addition_CPU_OpenMP
    addition_Cuda
    addition_Cuda_reduction
    addition_OpenCL
    addition_OpenCL_reduction)
   add_subdirectory(${d})
endforeach()


