cmake_minimum_required(VERSION 2.8.8)
set(ProjectName "Poisson_MPI_OpenMP_CoarseGrain")

project(${ProjectName})

if(POLICY CMP0054)
  cmake_policy(SET CMP0054 OLD)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif()

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7)
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
endif()

add_subdirectory(timer)

include_directories(
  timer
  )

find_package(MPI)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

add_definitions("-DUSE_MPI")
add_executable(
  ${ProjectName} 
  main.cxx
  parameters.cxx
  parameters.hxx
  values.cxx
  values.hxx
  scheme.cxx
  scheme.hxx
  )

target_link_libraries(
   ${ProjectName}
   timer
   ${MPI_CXX_LIBRARIES}
)

