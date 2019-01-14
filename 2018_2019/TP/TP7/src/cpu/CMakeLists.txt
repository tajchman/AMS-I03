include_directories(
  ${CMAKE_SOURCE_DIR}/common
  )

add_executable(image_cpu
  main.cxx
  rgb_to_grey.cxx
  )

target_link_libraries(image_cpu
  common
  png)

install(TARGETS image_cpu DESTINATION .)
