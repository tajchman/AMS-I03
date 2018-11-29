set(code "sinus_mpi")

add_definitions(-DOMPI_SKIP_MPICXX)
  
add_executable(${code} sinus.cxx ../sin.cxx)

target_link_libraries(${code}
  ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} util)

target_include_directories(${code}
  PRIVATE ${MPI_CXX_INCLUDE_PATH})

target_compile_options(${code}
  PRIVATE ${MPI_CXX_COMPILE_FLAGS})

install(TARGETS ${code} DESTINATION .)


