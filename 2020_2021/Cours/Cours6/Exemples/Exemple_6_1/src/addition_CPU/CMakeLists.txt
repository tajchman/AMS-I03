get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_executable(${CODE}
  main.cxx
  calcul_CPU.cxx
  ${CMAKE_SOURCE_DIR}/util/timer.cxx
  )


install(TARGETS ${CODE} DESTINATION .)
