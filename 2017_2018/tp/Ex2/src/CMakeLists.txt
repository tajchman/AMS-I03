cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

find_package(PAPI)

add_executable(Ex2
  util.cpp
  main.cpp
  )


