project(Ex2)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_RULE_MESSAGES OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Threads REQUIRED)
add_executable(ex_2_2.exe main.cxx init.cxx calcul.cxx)
target_link_libraries(ex_2_2.exe ${CMAKE_THREAD_LIBS_INIT})