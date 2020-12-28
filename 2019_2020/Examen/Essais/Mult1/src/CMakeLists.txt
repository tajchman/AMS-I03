cmake_minimum_required(VERSION 2.8)

get_filename_component(BASE_DIR ${CMAKE_SOURCE_DIR} PATH)
get_filename_component(VERSION ${BASE_DIR} NAME)

project(${VERSION})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(MPI)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
add_definitions("-DUSE_MPI")

include(FindOpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(
  util
  ${CMAKE_BUILD_DIR}
  )

add_executable(
  ${VERSION} 
  main.cxx
  matrix.cxx
  )

target_link_libraries(
   ${VERSION}
   util
   ${MPI_CXX_LIBRARIES}
)

install(TARGETS ${VERSION} DESTINATION .)

configure_file(version.hxx.in version.hxx @ONLY)

