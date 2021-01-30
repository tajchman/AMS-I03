
find_package(CUDAToolkit REQUIRED)

include_directories(
  .
  ${CUDA_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
 /usr/local/cuda/targets/x86_64-linux/include/
  )

add_executable(deviceQuery_cuda deviceQuery.cpp)

target_link_libraries(
  deviceQuery_cuda
  ${CUDA_CUDART}
  )

install(TARGETS deviceQuery_cuda DESTINATION .)


