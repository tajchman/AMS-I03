cmake_minimum_required(VERSION 2.8)

set(VERSION PoissonOpenCL)
project(${VERSION} LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCL REQUIRED)

include_directories(
  ${OpenCL_INCLUDE_DIRS}
  )

add_definitions(-DCL_TARGET_OPENCL_VERSION=${OpenCL_VERSION})
add_definitions(-DINSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")

set(OpenCL_VERSION ${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0)
  
include_directories(
  ${OpenCL_INCLUDE_DIRS}
  )

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
  opencl
  ${OPENCL_INCLUDE_DIRS}
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
    opencl/variation.cxx
    opencl/iteration.cxx
    opencl/dim.cxx
    opencl/values.cxx
    opencl/OpenCL.cxx
  )

add_executable(
  ${VERSION}
  ${SRCS}
  )

target_link_libraries(
  ${VERSION} 
  ${OpenCL_LIBRARIES}
  )

install(TARGETS ${VERSION} DESTINATION .)
install(FILES gray.cl smooth.cl sobel.cl DESTINATION .)


