cmake_minimum_required(VERSION 2.8)
project(TP5 C CXX CUDA)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../install
    CACHE PATH "chemin d'installation par defaut" FORCE)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP)

find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/util)

add_subdirectory(util)

foreach(d
    addition_CPU
    addition_CPU_OpenMP
    addition_GPU
    addition_GPU_reduction
    Pi_CPU
    Pi_GPU)
   add_subdirectory(${d})
endforeach()


