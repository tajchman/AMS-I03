enable_language(CXX CUDA)

find_package(CUDA)
if (CUDA_FOUND)
  
  set(CUDA_NVCC_FLAGS
    "${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets;-ccbin=${CMAKE_C_COMPILER}")

  include_directories(
    reduction
    )

  add_executable(eqn_gpu
    main.cxx
    timerGPU.cu
    calcul.cu
    laplacien.cu
    init.cu
    forces.cu
    variation.cu
    save.cu
    reduction/reduction_kernel.cu
    )
  target_compile_options(eqn_gpu PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

  install(TARGETS eqn_gpu DESTINATION .)

endif()


