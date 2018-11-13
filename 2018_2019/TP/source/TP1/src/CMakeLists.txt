cmake_minimum_required(VERSION 2.8)
project(TP1 CXX Fortran)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)

if (CMAKE_VERSION VERSION_LESS "3.1")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
  else ()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif ()
else ()
  set (CMAKE_CXX_STANDARD 11)
endif ()

add_subdirectory(timer)
add_subdirectory(matrice)
add_subdirectory(transposee)
add_subdirectory(valeur_propre)


