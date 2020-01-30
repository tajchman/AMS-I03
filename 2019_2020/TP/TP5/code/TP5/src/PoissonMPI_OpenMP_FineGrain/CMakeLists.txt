get_filename_component(VERSION ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_executable(
  ${VERSION} 
  main.cxx
  parameters.cxx
  parameters.hxx
  values.cxx
  values.hxx
  scheme.cxx
  scheme.hxx
  )

target_link_libraries(
   ${VERSION}
   util
   ${MPI_CXX_LIBRARIES}
)

configure_file(run_hybrid.sh.in run_hybrid.sh @ONLY)
install(TARGETS ${VERSION} DESTINATION .)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/run_hybrid.sh DESTINATION .)
install(FILES job_hybrid.in DESTINATION .)

