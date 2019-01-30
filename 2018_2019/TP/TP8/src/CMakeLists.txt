cmake_minimum_required( VERSION 2.8 )

set(CUDA_NVCC_FLAGS
  "${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets;-ccbin=${CMAKE_C_COMPILER}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ptxas-options=-v")
enable_language(CUDA)
enable_language(CXX)

include_directories(
  reduction
  )

add_executable(eqn_cpu
  main.cxx
  calcul.cxx
  )

add_executable(eqn_gpu
  main.cxx
  calcul.cu
  reduction/reduction_kernel.cu
  )

install(TARGETS eqn_cpu eqn_gpu DESTINATION .)

