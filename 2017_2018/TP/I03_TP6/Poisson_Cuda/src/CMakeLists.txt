cmake_minimum_required(VERSION 2.8.8)
set(ProjectName "Poisson_Cuda")

project(${ProjectName} LANGUAGES C CXX CUDA)


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

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets")

include_directories(
  timer
  )

add_executable(
  ${ProjectName} 
  main.cxx
  AbstractScheme.cxx
  AbstractScheme.hxx
  AbstractValues.cxx
  AbstractValues.cxx
  AbstractParameters.cxx
  AbstractParameters.hxx
  CpuScheme.cxx
  CpuScheme.hxx
  GpuScheme.cu
  GpuScheme.hxx
  CpuValues.cxx
  CpuValues.hxx
  GpuValues.cu
  GpuValues.hxx
  CpuParameters.cxx
  CpuParameters.hxx
  GpuParameters.cu
  GpuParameters.hxx
  )

set_property(TARGET ${ProjectName}
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
             
target_link_libraries(
   ${ProjectName}
   timer
   cuda
   ${CUDA_LIBRARIES}
)

