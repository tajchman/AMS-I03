set(CODE addition_Seq)
project(${CODE} LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_compile_options(-DDEBUG)
endif()

add_executable(${CODE}
  main.cxx
  calcul_Seq.cxx
  timer.cxx
  )

install(TARGETS ${CODE} DESTINATION .)
