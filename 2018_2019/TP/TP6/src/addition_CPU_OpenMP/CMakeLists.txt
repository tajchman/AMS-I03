get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (OPENMP_FOUND)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
  
add_executable(${CODE}
  main.cxx
  calcul_CPU.cxx
  )

target_link_libraries(${CODE}
  util
  )

install(TARGETS ${CODE} DESTINATION .)
