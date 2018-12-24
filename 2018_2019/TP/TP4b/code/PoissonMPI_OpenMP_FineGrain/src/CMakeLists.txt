cmake_minimum_required(VERSION 2.8)

get_filename_component(BASE_DIR ${CMAKE_SOURCE_DIR} PATH)
get_filename_component(VERSION ${BASE_DIR} NAME)

project(${VERSION})

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE "Debug" CACHE
      STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${BASE}/install
      CACHE PATH "chemin d'installation par defaut" FORCE)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FindOpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(
  util
  )

find_package(MPI)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
include_directories(util)

add_definitions("-DUSE_MPI")
add_executable(
  ${VERSION} 
  main.cxx
  parameters.cxx
  parameters.hxx
  values.cxx
  values.hxx
  scheme.cxx
  scheme.hxx
  )

target_link_libraries(
   ${VERSION}
   util
   ${MPI_CXX_LIBRARIES}
)

install(TARGETS ${VERSION} DESTINATION .)
