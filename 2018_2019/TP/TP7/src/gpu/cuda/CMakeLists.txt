set(CODE image_gpu_cuda)

enable_language(CUDA)
set(CUDA_NVCC_FLAGS
  "${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets;-ccbin=${CMAKE_C_COMPILER}")

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

