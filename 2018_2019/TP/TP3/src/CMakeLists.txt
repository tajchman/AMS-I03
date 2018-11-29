cmake_minimum_required(VERSION 2.8)
project(TP3 CXX)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../install
    CACHE PATH "chemin d'installation par defaut" FORCE)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (CMAKE_VERSION VERSION_LESS "3.1")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
  else ()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif ()
else ()
  set (CMAKE_CXX_STANDARD 11)
  set (CMAKE_INSTALL_MESSAGE LAZY)
endif ()

include(FindOpenMP)

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/util)

add_subdirectory(util)

foreach(d seq fine coarse_1 coarse_2 tasks)
   add_subdirectory(sinus_${d})
endforeach()
add_subdirectory(exemple_tasks)


