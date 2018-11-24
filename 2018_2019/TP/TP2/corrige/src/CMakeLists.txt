cmake_minimum_required(VERSION 2.8)
project(TP2 CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (CMAKE_VERSION VERSION_LESS "3.1")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
  else ()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif ()
else ()
  set (CMAKE_CXX_STANDARD 11)
endif ()

set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_DEBUG} -pg") 
set(CMAKE_Fortran_FLAGS_PROFILE "${CMAKE_Fortran_FLAGS_DEBUG} -pg") 
set(CMAKE_EXE_LINKER_FLAGS_PROFILE "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -pg")

include(FindOpenMP)

include_directories(${CMAKE_SOURCE_DIR}/util)

add_subdirectory(util)

foreach(d sinus_seq sinus_openmp_1 sinus_std_threads)
   add_subdirectory(${d})
endforeach()


