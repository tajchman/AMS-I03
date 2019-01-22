find_package(CUDA)

if (CUDA_FOUND)
  add_subdirectory(cuda)
else()
  message(STATUS "CUDA not found : cannot compile CUDA version")
endif()

find_package(OpenCL)

if (OpenCL_FOUND)
  add_subdirectory(opencl)
else()
  message(STATUS "OpenCL not found : cannot compile OpenCL version")
endif()
 
