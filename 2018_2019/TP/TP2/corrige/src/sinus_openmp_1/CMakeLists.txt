set(code "sinus_openmp_1")

set_source_files_properties(sinus.cxx
  PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  
add_executable(${code} sinus.cxx)
target_link_libraries(${code} util ${OpenMP_CXX_LIBRARIES})

install(TARGETS ${code} DESTINATION .)

