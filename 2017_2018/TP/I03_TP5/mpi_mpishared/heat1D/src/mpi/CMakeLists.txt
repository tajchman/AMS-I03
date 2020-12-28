
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
add_definitions("-DUSE_MPI")

add_executable(
  heat1D_mpi
  heat1D_mpi.c
  )

target_link_libraries(
   heat1D_mpi
   ${MPI_CXX_LIBRARIES}
)


