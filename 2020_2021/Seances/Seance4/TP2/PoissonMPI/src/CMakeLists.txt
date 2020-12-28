cmake_minimum_required(VERSION 2.8)

project(Tasks)

find_package(MPI REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_compile_options(-DDEBUG)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qdiag-disable:1786,3180")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=1786,3180")
  endif()
endif()

add_executable(
  poisson_seq.exe
  main_seq.cxx
  #memory_used.cxx
  arguments.cxx)

include_directories(
  ${MPI_INCLUDES}
)

add_executable(
  poisson_mpi.exe
  main_mpi.cxx
  #memory_used.cxx
  arguments.cxx)

install(TARGETS poisson_seq.exe poisson_mpi.exe DESTINATION .)
 
