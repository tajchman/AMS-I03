project(NN)
cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)

set(PNG_BUILD_ZLIB ${CMAKE_BINARY_DIR}/libs/zlib-1.2.11)
set(ZLIB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/libs/zlib-1.2.11)
set(ZLIB_LIBRARY z)
set(PNG_PNG_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/libs/libpng-1.6.37)
set(PNG_LIBDIR ${CMAKE_BINARY_DIR}/libs/libpng-1.6.37)
set(PNG_LIBRARIES png z)

link_directories(${PNG_BUILD_ZLIB} ${PNG_LIBDIR})

include_directories(
  ${PNG_PNG_INCLUDE_DIR}
  )

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

add_subdirectory(libs)
add_executable(nn_CPU main.cxx io_png.cxx reseau.cxx input.cxx)

target_link_libraries(nn_CPU
  ${PNG_LIBRARIES}
  )

install(TARGETS nn_CPU DESTINATION .)

