set(CODE addition_MPI)
project(${CODE} LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_compile_options(-DDEBUG)
endif()

find_package(MPI)

include_directories(
  ${MPI_INCLUDE_PATH}
)

add_executable(${CODE}
  main.cxx
  Calcul_MPI.cxx
  timer.cxx
  )

target_link_libraries(${CODE} ${MPI_LIBRARIES})

install(TARGETS ${CODE} DESTINATION .)
