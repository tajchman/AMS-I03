set(CODE image_gpu_cuda)

find_package(CUDA)

if (CUDA_FOUND)

  message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
  enable_language(CUDA)

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

else()
  message(STATUS "CUDA not found : cannot compile CUDA version")
endif()
