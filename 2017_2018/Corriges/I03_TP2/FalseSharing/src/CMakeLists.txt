cmake_minimum_required(VERSION 2.8.8)
project(FalseSharing)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
endif()

find_package(OpenMP)
 
add_subdirectory(sequentiel)
add_subdirectory(multithreads_1)
add_subdirectory(multithreads_2)
add_subdirectory(multithreads_3)
add_subdirectory(multithreads_4)

