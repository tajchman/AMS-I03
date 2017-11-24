cmake_minimum_required(VERSION 2.8.8)
project(POISSON)

message(STATUS "compiler: ${CMAKE_CXX_COMPILER_ID}")
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Clang detected, adding compile flags")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "GCC detected, adding compile flags")
    add_definitions(-Wall -Wextra)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    message(STATUS "Intel detected, adding compile flags")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(STATUS "Visual C/C++ detected, adding compile flags")
    add_definitions(/D_CRT_SECURE_NO_WARNINGS)
endif()

add_executable(PoissonSeq main.cxx parameters.cxx values.cxx scheme.cxx timer.cxx time_unix.c time_windows.c)

