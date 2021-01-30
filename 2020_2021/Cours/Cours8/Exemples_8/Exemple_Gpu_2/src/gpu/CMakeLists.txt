if (ENABLE_CUDA)
  add_subdirectory(cuda)
endif()

if (ENABLE_OPENCL)
  add_subdirectory(opencl)
endif()

