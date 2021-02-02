cmake_minimum_required(VERSION 2.8)

set(VERSION PoissonCuda)
project(${VERSION} LANGUAGES CXX CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(CUDAToolkit)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 11)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
add_definitions("-Wno-deprecated-gpu-targets")

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_compile_options(-DDEBUG)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qdiag-disable:1786,3180,10148")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=1786,3180,10148")
  endif()
endif()

include_directories(
  util
  ${CMAKE_BUILD_DIR}
  cuda
  ${CUDAToolkit_INCLUDE_DIRS}
  )

configure_file(version.hxx.in version.hxx @ONLY)

set(SRCS
    main.cxx
    parameters.cxx
    scheme.cxx
    values.cxx
    util/arguments.cxx
    util/os.cxx
    util/timer.cxx
    cuda/variation.cu
    cuda/iteration.cu
    cuda/dim.cu
    cuda/values.cu
    cuda/user.cu
    cuda/Cuda.cu
  )

add_executable(
  ${VERSION}
  ${SRCS}
  )

set_target_properties(${VERSION}
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(
  ${VERSION} 
  ${CUDA_CUDART}
  )

install(TARGETS ${VERSION} DESTINATION .)

