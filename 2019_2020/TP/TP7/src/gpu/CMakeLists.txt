find_package(CUDA)

if (CUDA_FOUND)
  add_subdirectory(cuda)
else()
  message(STATUS "CUDA not found : cannot compile CUDA version")
endif()

