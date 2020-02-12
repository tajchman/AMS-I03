
set_source_files_properties(main_tasks.cxx
  PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

add_executable(exemple_seq main_seq.cxx ../sin.cxx)
target_link_libraries(exemple_seq util)
add_executable(exemple_tasks main_tasks.cxx ../sin.cxx)
target_link_libraries(exemple_tasks util ${OpenMP_CXX_LIBRARIES})

install(TARGETS exemple_seq exemple_tasks DESTINATION .)


