set(CODE image_cpu)

add_executable(${CODE}
  ${CMAKE_SOURCE_DIR}/main.cxx
  process.cxx
  )

target_link_libraries(${CODE}
  util
  ${PNG_LIBRARIES}
  )

install(TARGETS ${CODE} DESTINATION .)
