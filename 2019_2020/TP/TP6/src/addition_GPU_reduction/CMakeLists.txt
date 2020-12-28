get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

include_directories(
  reduction
  )

add_executable(${CODE}
  main.cxx
  calcul_GPU.cu
  reduction/reduction_kernel.cu)

target_link_libraries(${CODE}
  util
  )

install(TARGETS ${CODE} DESTINATION .)
