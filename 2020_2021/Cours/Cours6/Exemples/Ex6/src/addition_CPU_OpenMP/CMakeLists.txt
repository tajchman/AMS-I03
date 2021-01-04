cmake_minimum_required(VERSION 3.18)

get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${CODE} C CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(util)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
  
add_executable(${CODE}
  main.cxx
  calcul_CPU.cxx
  util/timer.cxx
  )

install(TARGETS ${CODE} DESTINATION .)
