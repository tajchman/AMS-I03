cmake_minimum_required(VERSION 3.17)
project(query LANGUAGES C CXX CUDA)

find_package(CUDAToolkit)

include_directories(
  .
  ${CUDA_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
  )

add_executable(deviceQuery deviceQuery.cpp)

target_link_libraries(
  deviceQuery
  ${CUDA_CUDART}
  )

install(TARGETS deviceQuery DESTINATION .)


