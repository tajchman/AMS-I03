set(VERSION gpu_cuda)
set(CODE image_${VERSION})

configure_file(../../version.hxx.in version.hxx @ONLY)

find_package(CUDA)

if (CUDA_FOUND)
  enable_language(CUDA)

  set(CMAKE_CUDA_ARCHITECTURES OFF)
  cuda_select_nvcc_arch_flags(cuda_archs Auto)
  list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
  set(CUDA_NVCC_FLAGS
    "${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets;-ccbin=${CMAKE_C_COMPILER}")

  add_executable(${CODE}
    ${CMAKE_SOURCE_DIR}/main.cxx
    process.cu
  )

  target_link_libraries(${CODE}
    util
    ${PNG_LIBRARIES}
    ${CUDA_CUDART}
  )

  install(TARGETS ${CODE} DESTINATION .)

endif()

