set_source_files_properties(sinus.cxx
  PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

set_source_files_properties(sinus_2.cxx
  PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

add_executable(sinus_fine sinus.cxx ../sin.cxx)
target_link_libraries(sinus_fine util ${OpenMP_CXX_LIBRARIES})

install(TARGETS sinus_fine DESTINATION .)

add_executable(sinus_fine_2 sinus_2.cxx ../sin.cxx)
target_link_libraries(sinus_fine_2 util ${OpenMP_CXX_LIBRARIES})

install(TARGETS sinus_fine_2  DESTINATION .)


