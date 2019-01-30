cmake_minimum_required( VERSION 2.8 )

set(CUDA_NVCC_FLAGS
  "${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets;-ccbin=${CMAKE_C_COMPILER}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ptxas-options=-v")
enable_language(CUDA)
enable_language(CXX)


add_executable(eqn_cpu
  main.cxx
  calcul.cxx
  )

if (CUDA_FOUND)
  
  set(GPU_CODE eqn_gpu) 

  include_directories(
    reduction
    )

  add_executable(${GPU_CODE}
    main.cxx
    calcul.cu
    reduction/reduction_kernel.cu
    )

endif()

install(TARGETS eqn_cpu ${GPU_CODE} DESTINATION .)

