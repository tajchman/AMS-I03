
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
add_definitions("-DUSE_MPI")

add_executable(
  heat1D_mpishared
  heat1D_mpishared.c
  )

target_link_libraries(
   heat1D_mpishared
   ${MPI_CXX_LIBRARIES}
)


