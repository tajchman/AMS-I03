project(Ex_6_1 LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(util)

set(DIRS 
    addition_CPU
    addition_CPU_OpenMP
)

option(ENABLE_CUDA "Enable Cuda" OFF)
if (ENABLE_CUDA)
  set(DIRS ${DIRS}
    addition_Cuda
    addition_Cuda_reduction)
endif()

option(ENABLE_OPENCL "Enable OpenCL" OFF)
if (ENABLE_OPENCL)
  set(DIRS ${DIRS}
    addition_OpenCL
    addition_OpenCL_reduction)
endif()


foreach(d ${DIRS})
   add_subdirectory(${d})
endforeach()


