
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

configure_file(run_mpi.sh.in run_mpi.sh @ONLY)
install(TARGETS ${VERSION} DESTINATION .)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/run_mpi.sh DESTINATION .)
install(FILES job_mpi.in DESTINATION .)

