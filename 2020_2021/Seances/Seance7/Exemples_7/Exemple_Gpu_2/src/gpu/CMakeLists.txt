if (ENABLE_CUDA)

  enable_language(CUDA)
  find_package(CUDA)

  if (CUDA_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES 37 50 75)
    add_subdirectory(cuda)
  else()
    message(STATUS "CUDA not found : cannot compile CUDA version")
  endif()

endif()

if (ENABLE_OPENCL)

  cmake_host_system_information(RESULT machine QUERY HOSTNAME)
  if (${machine} STREQUAL "rhum.ensta.fr")
    set(OpenCL_FOUND "ON")
    set(OpenCL_INCLUDE_DIR 
       "/usr/local/cuda-9.0/targets/x86_64-linux/include")
    set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
    set(OpenCL_LIBRARY
       "/usr/local/cuda-9.0/targets/x86_64-linux/lib/libOpenCL.so.1.0.0")
    set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
    set(OpenCL_VERSION 120)
  else()
    find_package(OpenCL)
  endif()
  
  if (OpenCL_FOUND)
      add_subdirectory(opencl)
  endif()

endif()

