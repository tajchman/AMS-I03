get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set_source_files_properties(sinus.cxx
  PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  
add_executable(${CODE} 
  sinus.cxx
  ../sin.cxx
  charge.cxx
)

target_link_libraries(${CODE}
  util
  ${OpenMP_CXX_LIBRARIES})

install(TARGETS ${CODE} DESTINATION .)


