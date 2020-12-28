cmake_minimum_required(VERSION 2.8)

project(STDTHREADS)

set(CMAKE_RULE_MESSAGES OFF)
#set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_compile_options(-DDEBUG)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qdiag-disable:1786,3180")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=1786,3180")
  endif()
endif()

add_executable(
  exemple_par.exe 
  main_par.cxx
  arguments.cxx)

target_link_libraries(
  exemple_par.exe
  ${CMAKE_THREAD_LIBS_INIT})

add_executable(
  exemple_par2.exe 
  main_par2.cxx
  arguments.cxx)

target_link_libraries(
  exemple_par2.exe
  ${CMAKE_THREAD_LIBS_INIT})

add_executable(
  exemple_seq.exe 
  main_seq.cxx
  arguments.cxx)

install(TARGETS exemple_seq.exe exemple_par.exe exemple_par2.exe DESTINATION .)


