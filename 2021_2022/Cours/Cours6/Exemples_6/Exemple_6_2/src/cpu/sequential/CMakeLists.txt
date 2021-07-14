set(CODE image_cpu)

add_executable(${CODE}
  ${CMAKE_SOURCE_DIR}/main.cxx
  process.cxx
  ${CMAKE_SOURCE_DIR}/util/io_png.cxx
  )

target_link_libraries(${CODE}
  ${PNG_LIBRARIES}
  )

install(TARGETS ${CODE} DESTINATION .)
