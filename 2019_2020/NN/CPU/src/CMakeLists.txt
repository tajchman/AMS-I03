cmake_minimum_required(VERSION 3.5)

find_package(PNG)

include_directories(
  ${PNG_PNG_INCLUDE_DIR}
  )

add_executable(nn_CPU main.cxx io_png.cxx reseau.cxx input.cxx)

target_link_libraries(nn_CPU
  ${PNG_LIBRARIES}
  )

install(TARGETS nn_CPU DESTINATION .)

