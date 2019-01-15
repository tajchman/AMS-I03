include_directories(
  ${CMAKE_SOURCE_DIR}/common
  )

add_executable(image_cpu
  main.cxx
  process.cxx
  )

target_link_libraries(image_cpu
  common
  ${PNG_LIBRARIES}
  )

install(TARGETS image_cpu DESTINATION .)
