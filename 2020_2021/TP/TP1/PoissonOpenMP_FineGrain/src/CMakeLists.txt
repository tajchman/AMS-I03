cmake_minimum_required(VERSION 2.8)

get_filename_component(BASE_DIR ${CMAKE_SOURCE_DIR} PATH)
get_filename_component(VERSION ${BASE_DIR} NAME)

project(${VERSION})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_compile_options(--inline)
endif()

include(FindOpenMP)

add_subdirectory(util)

include_directories(
  util
  ${CMAKE_BUILD_DIR}
  )

set(REFERENCE PoissonSeq)

foreach (CODE ${VERSION} ${REFERENCE})

  add_executable(
    ${CODE} 
    main.cxx
    parameters.cxx
    parameters.hxx
    values.cxx
    values.hxx
    scheme.cxx
    scheme.hxx
  )

  target_link_libraries(
    ${CODE}
    util
  )

  install(TARGETS ${CODE} DESTINATION .)

endforeach()

target_compile_options(${VERSION} PUBLIC ${OpenMP_CXX_FLAGS})
target_link_libraries(${VERSION} ${OpenMP_CXX_LIBRARIES})

configure_file(version.hxx.in version.hxx @ONLY)
