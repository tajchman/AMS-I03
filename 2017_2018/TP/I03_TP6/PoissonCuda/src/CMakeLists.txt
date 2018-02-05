cmake_minimum_required(VERSION 2.8.8)
project(POISSON LANGUAGES C CXX CUDA)

if(POLICY CMP0054)
  cmake_policy(SET CMP0054 OLD)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif()

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7)
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
endif()

find_package(CUDA)

add_subdirectory(timer)

include_directories(
  timer
  )

cuda_add_executable(
  PoissonCuda 
  main.cxx
  parameters.cxx
  parameters.hxx
  values.cxx
  values.hxx
  scheme.cxx
  scheme.hxx
  gpu_scheme.cu
  gpu_scheme.hxx
  gpu_values.cu
  gpu_values.hxx
  )

set_property(TARGET PoissonCuda 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
             
target_link_libraries(
   PoissonCuda
   timer
   cuda
   ${CUDA_LIBRARIES}
)

