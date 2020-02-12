set(CODE image_cpu)

include_directories(
  ${CMAKE_SOURCE_DIR}/common
  ${CMAKE_SOURCE_DIR}/util
  )

add_executable(${CODE}
  ${CMAKE_SOURCE_DIR}/common/main.cxx
  process.cxx
  )

target_link_libraries(${CODE}
  common
  ${PNG_LIBRARIES}
  )

install(TARGETS ${CODE} DESTINATION .)
