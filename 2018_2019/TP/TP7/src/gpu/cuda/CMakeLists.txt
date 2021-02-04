set(CODE image_gpu_cuda)

set(CUDA_NVCC_FLAGS
  "${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets;-ccbin=${CMAKE_C_COMPILER}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ptxas-options=-v")
enable_language(CUDA)

add_subdirectory(deviceQuery)

include_directories(
  ${CMAKE_SOURCE_DIR}/common
  ${CMAKE_SOURCE_DIR}/util
  )

add_executable(${CODE}
  ${CMAKE_SOURCE_DIR}/common/main.cxx
  process.cu
  )

target_link_libraries(${CODE}
  common
  ${PNG_LIBRARIES}
  )

install(TARGETS ${CODE} DESTINATION .)


