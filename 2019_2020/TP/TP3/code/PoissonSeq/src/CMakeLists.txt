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


set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(
  util
  ${CMAKE_BUILD_DIR}
  )

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

