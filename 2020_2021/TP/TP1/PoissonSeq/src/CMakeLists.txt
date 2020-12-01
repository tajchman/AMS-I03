cmake_minimum_required(VERSION 2.8)

project(Seq)
set(VERSION PoissonSeq)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_compile_options(-DDEBUG --inline)
endif()

include_directories(
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
    util/keyPress.cxx
    util/os.cxx
    util/pause.cxx
    util/timer.cxx
  )

add_executable(
  ${VERSION} 
  ${SRCS})

target_link_libraries(
  ${VERSION}
  util)

install(TARGETS ${VERSION} DESTINATION .)

