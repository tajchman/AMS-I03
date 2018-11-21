set(code "moyenne_openmp_1")

set_source_files_properties(${code}.cxx
  PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  
add_executable(${code} ${code}.cxx)
target_link_libraries(${code} util ${OpenMP_CXX_LIBRARIES})

install(TARGETS ${code} DESTINATION .)
