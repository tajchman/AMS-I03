set(CODE image_gpu_cuda)

add_definitions("-Wno-deprecated-gpu-targets")
set(CUDA_NVCC_FLAGS
  "${CUDA_NVCC_FLAGS};-ccbin=${CMAKE_C_COMPILER}")

set(CMAKE_CUDA_ARCHITECTURES 37 50 75)

add_executable(${CODE}
  ${CMAKE_SOURCE_DIR}/main.cxx
  process.cu
  ${CMAKE_SOURCE_DIR}/util/io_png.cxx
  )

target_link_libraries(${CODE}
  ${PNG_LIBRARIES}
  )

install(TARGETS ${CODE} DESTINATION .)

