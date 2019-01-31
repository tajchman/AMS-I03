enable_language(CXX CUDA)

find_package(CUDA)
if (CUDA_FOUND)
  
  set(CUDA_NVCC_FLAGS
    "${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets;-ccbin=${CMAKE_C_COMPILER}")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ptxas-options=-v")

  add_executable(stride
    stride.cu
    )

  install(TARGETS stride DESTINATION .)

endif()


