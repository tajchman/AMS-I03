get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_executable(${CODE} main.cxx calcul_CPU.cxx calcul_GPU.cu)

target_link_libraries(${CODE}
  util
  )

install(TARGETS ${CODE} DESTINATION .)
