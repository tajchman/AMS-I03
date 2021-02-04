cmake_minimum_required(VERSION 2.8)

set(VERSION PoissonSeq)
project(${VERSION} LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (CMAKE_VERSION VERSION_LESS "3.1")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
  endif ()
else()
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()

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
  )

configure_file(version.hxx.in version.hxx @ONLY)

set(SRCS
    main.cxx
    parameters.cxx
    scheme.cxx
    values.cxx
    user.cxx
    util/arguments.cxx
    util/os.cxx
    util/timer.cxx
  )

add_executable(
  ${VERSION}
  ${SRCS}
  )

install(TARGETS ${VERSION} DESTINATION .)

