project(NN)
cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)

if (SYSTEM_PNG)
  find_package(PNG)
else()
  get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
  get_filename_component(PARENT_DIR ${PARENT_DIR} DIRECTORY)
  set(LIBS_DIR ${PARENT_DIR}/Libs/install/${CMAKE_BUILD_TYPE})
  set(PNG_PNG_INCLUDE_DIR ${LIBS_DIR}/include)
  link_directories(${LIBS_DIR}/lib
                   ${LIBS_DIR}/lib64)
  set(PNG_LIBRARIES libpng16d zlib)
  message(STATUS  ${PNG_PNG_INCLUDE_DIR})
endif()

include_directories(
  ${PNG_PNG_INCLUDE_DIR}
  )

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

add_executable(nn_CPU main.cxx io_png.cxx reseau.cxx input.cxx)

target_link_libraries(nn_CPU
  ${PNG_LIBRARIES}
  )

install(TARGETS nn_CPU DESTINATION .)

