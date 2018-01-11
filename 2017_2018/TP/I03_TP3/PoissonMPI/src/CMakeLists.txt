cmake_minimum_required(VERSION 2.8.8)
project(POISSON)

if(POLICY CMP0054)
  cmake_policy(SET CMP0054 OLD)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif()

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
endif()

add_subdirectory(timer)

include_directories(
  timer
  )

find_package(MPI)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

add_executable(
  PoissonMPI
  main.cxx
  parameters.cxx
  values.cxx
  scheme.cxx
  )

target_link_libraries(
   PoissonMPI
   timer
   ${MPI_CXX_LIBRARIES}
)
