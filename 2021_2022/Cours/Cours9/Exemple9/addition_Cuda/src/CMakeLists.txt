set(CODE addition_Cuda)
project(${CODE} LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_compile_options(-DDEBUG)
endif()

find_package(CUDA)

if (CUDA_FOUND)
  enable_language(CUDA)

  set(CMAKE_CUDA_ARCHITECTURES OFF)
  cuda_select_nvcc_arch_flags(cuda_archs Auto)
  list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

  include_directories(
    ${CUDA_TOOLKIT_INCLUDE}
    )

  add_executable(${CODE}
    main.cxx
    Calcul_Cuda.cu
    reduction_kernel.cu
    timer.cxx
    )

  install(TARGETS ${CODE} DESTINATION .)

endif()

