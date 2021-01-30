get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

find_package(CUDA)

if (CUDA_FOUND)
  enable_language(CUDA)

  set(CMAKE_CUDA_ARCHITECTURES OFF)
  cuda_select_nvcc_arch_flags(cuda_archs Auto)
  list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

  include_directories(
    reduction
    ${CUDA_TOOLKIT_INCLUDE}
    )

  add_executable(${CODE}
    main.cxx
    Calcul_Cuda.cu
   reduction/reduction_kernel.cu
    ${CMAKE_SOURCE_DIR}/util/timer.cxx
    )

  install(TARGETS ${CODE} DESTINATION .)

endif()

