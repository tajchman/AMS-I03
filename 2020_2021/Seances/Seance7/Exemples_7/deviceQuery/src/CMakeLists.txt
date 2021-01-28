cmake_minimum_required(VERSION 3.05)
project(query LANGUAGES C CXX CUDA)

find_package(CUDAToolkit)

include_directories(
  .
  ${CUDA_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
 /usr/local/cuda/targets/x86_64-linux/include/
  )

add_executable(deviceQuery deviceQuery.cpp)

target_link_libraries(
  deviceQuery
  ${CUDA_CUDART}
  )

install(TARGETS deviceQuery DESTINATION .)


