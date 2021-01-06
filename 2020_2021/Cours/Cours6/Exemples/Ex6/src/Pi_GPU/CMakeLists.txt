cmake_minimum_required(VERSION 3.18)

get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${CODE} C CXX CUDA)

include_directories(util)

add_executable(${CODE}
  main.cxx
  calcul_Pi.cu)

install(TARGETS ${CODE} DESTINATION .)
