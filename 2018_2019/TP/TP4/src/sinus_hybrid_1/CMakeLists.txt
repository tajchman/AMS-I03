set(code "sinus_hybrid_1")

add_definitions(-DOMPI_SKIP_MPICXX)
  
add_executable(${code} sinus.cxx ../sin.cxx)

target_link_libraries(${code}
  util
  ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS}
  ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_LIBRARIES})

target_include_directories(${code}
  PRIVATE ${MPI_CXX_INCLUDE_PATH})

target_compile_options(${code}
  PRIVATE ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})

install(TARGETS ${code} DESTINATION .)


