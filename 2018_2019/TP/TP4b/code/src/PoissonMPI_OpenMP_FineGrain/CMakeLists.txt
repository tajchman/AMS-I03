get_filename_component(VERSION ${CMAKE_CURRENT_SOURCE_DIR} NAME)

include(FindOpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

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

