set(VERSION cpu)
set(CODE image_${VERSION})

configure_file(../../version.hxx.in version.hxx @ONLY)

add_executable(${CODE}
  ${CMAKE_SOURCE_DIR}/main.cxx
  process.cxx
  )

target_link_libraries(${CODE}
  util
  ${PNG_LIBRARIES}
  )

install(TARGETS ${CODE} DESTINATION .)
