cmake_minimum_required(VERSION 3.16)
project(query LANGUAGES CUDA)
include_directories(
  .
  ${CUDA_INCLUDE_DIRS}
  )

add_executable(deviceQuery deviceQuery.cpp)
target_link_libraries(deviceQuery
  ${CUDA_LIBRARIES}
  )

install(TARGETS deviceQuery DESTINATION .)


