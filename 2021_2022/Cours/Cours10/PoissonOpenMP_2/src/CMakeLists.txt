cmake_minimum_required(VERSION 2.8)

if (ENABLE_OPENMP)
  set(VERSION PoissonOpenMP)
else()
  set(VERSION PoissonSeq)
endif()

project(${VERSION} LANGUAGES CXX)

if (ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

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

