cmake_minimum_required(VERSION 3.05)
project(query LANGUAGES C CXX)

cmake_minimum_required(VERSION 3.1)

option(ENABLE_CUDA "Enable Cuda" OFF)
if(ENABLE_CUDA)
  add_subdirectory(cuda)
endif()

option(ENABLE_OPENCL "Enable OpenCL" OFF)
if(ENABLE_OPENCL)
  add_subdirectory(opencl)
endif()

install(FILES dummy DESTINATION .)
