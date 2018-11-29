get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_definitions(-DOMPI_SKIP_MPICXX)
  
add_executable(${CODE} sinus.cxx ../sin.cxx)

target_link_libraries(${CODE}
  util
  ${MPI_CXX_LINK_FLAGS}
  ${MPI_CXX_LIBRARIES} 
  ${OpenMP_CXX_FLAGS}
  ${OpenMP_CXX_LIBRARIES})

target_include_directories(${CODE}
  PRIVATE ${MPI_CXX_INCLUDE_PATH})

target_compile_options(${CODE}
  PRIVATE 
  ${MPI_CXX_COMPILE_FLAGS}
  ${OpenMP_CXX_FLAGS})

install(TARGETS ${CODE} DESTINATION .)


