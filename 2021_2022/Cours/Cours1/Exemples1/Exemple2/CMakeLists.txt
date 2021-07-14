project(Ex2)
cmake_minimum_required(VERSION 3.08)

set(CMAKE_RULE_MESSAGES OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

if (MESURE)
   add_compile_options(-DMESURE)
endif()

add_executable(ex_1_2.exe main.cxx init.cxx calcul.cxx)
