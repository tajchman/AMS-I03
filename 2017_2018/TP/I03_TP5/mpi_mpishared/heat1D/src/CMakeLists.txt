cmake_minimum_required(VERSION 2.8.8)
project(HEAT1D)

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

find_package(MPI)

add_subdirectory(sequentiel)
add_subdirectory(mpi)
add_subdirectory(mpi_mpishared)

