cmake_minimum_required(VERSION 2.8)
project(TP5 C CXX CUDA)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../install
    CACHE PATH "chemin d'installation par defaut" FORCE)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/util)

add_subdirectory(util)

foreach(d 1 2)
   add_subdirectory(addition_${d})
endforeach()


