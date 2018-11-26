cmake_minimum_required(VERSION 2.8)
project(POISSON)

get_filename_component(DIR get_file${CMAKE_SOURCE_DIR} NAME)
message(${DIR})

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE "Debug" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

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
endif ()

include(FindOpenMP)
if (OPENMP_FOUND)
  set (CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS 
    "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Debug")
endif()

add_subdirectory(timer)

include_directories(
  timer
  )

add_executable(
  PoissonOpenMP 
  main.cxx
  parameters.cxx
  parameters.hxx
  values.cxx
  values.hxx
  scheme.cxx
  scheme.hxx
  )

target_link_libraries(
   PoissonOpenMP
   timer
)


