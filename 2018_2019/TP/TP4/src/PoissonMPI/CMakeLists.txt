
find_package(MPI)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

add_definitions("-DUSE_MPI")
add_executable(
  PoissonMPI 
  main.cxx
  parameters.cxx
  parameters.hxx
  values.cxx
  values.hxx
  scheme.cxx
  scheme.hxx
  memory_used.c
  )

target_link_libraries(
   PoissonMPI
   util
   ${MPI_CXX_LIBRARIES}
)

install(TARGETS PoissonMPI DESTINATION .)

