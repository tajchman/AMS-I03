project(Ex_6_1 C CXX CUDA)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP)

find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

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


