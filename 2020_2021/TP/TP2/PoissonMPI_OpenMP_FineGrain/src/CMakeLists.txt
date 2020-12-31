cmake_minimum_required(VERSION 2.8)

project(PoissonMPI_FineGrain)
set(VERSION PoissonMPI_FineGrain)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

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

include_directories(
  ${MPI_CXX_INCLUDE_DIRS}
  util
  ${CMAKE_BUILD_DIR}
)

configure_file(version.hxx.in version.hxx @ONLY)

set(SRCS
  main.cxx
  parameters.cxx
  values.cxx
  scheme.cxx
  util/arguments.cxx
  util/os.cxx
  util/pause.cxx
  util/timer.cxx
  )

add_executable(
  ${VERSION}.exe 
  ${SRCS}
  )

target_link_libraries(
    ${VERSION}.exe 
    ${MPI_mpi_cxx_LIBRARY} 
    ${MPI_mpi_LIBRARY} 
    ${OpenMP_CXX_LIBRARIES}
)

install(TARGETS ${VERSION}.exe DESTINATION .)
 
