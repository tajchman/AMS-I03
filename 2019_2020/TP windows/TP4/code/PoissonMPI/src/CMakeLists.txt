cmake_minimum_required(VERSION 2.8)

get_filename_component(BASE_DIR ${CMAKE_SOURCE_DIR} PATH)
get_filename_component(VERSION ${BASE_DIR} NAME)

project(${VERSION})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(MPI)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

include_directories(
  util
  ${CMAKE_BUILD_DIR}
  )

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
