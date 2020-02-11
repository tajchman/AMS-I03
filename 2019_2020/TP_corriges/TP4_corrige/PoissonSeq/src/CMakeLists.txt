cmake_minimum_required(VERSION 2.8)

get_filename_component(BASE_DIR ${CMAKE_SOURCE_DIR} PATH)
get_filename_component(VERSION ${BASE_DIR} NAME)

project(${VERSION})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(
  util
  ${CMAKE_BUILD_DIR}
  )

add_subdirectory(util)

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
)

install(TARGETS ${VERSION} DESTINATION .)

configure_file(version.hxx.in version.hxx @ONLY)
message(STATUS ${CMAKE_INSTALL_PREFIX})

