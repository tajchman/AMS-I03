get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_definitions(
    -DOMPI_SKIP_MPICXX
    ${MPI_CXX_COMPILE_FLAGS}
    ${OpenMP_CXX_FLAGS} 
   )
  
add_executable(${CODE} 
  sinus.cxx
  ../sin.cxx
)

target_link_libraries(${CODE}
  util
  ${MPI_CXX_LIBRARIES}
  ${OpenMP_CXX_LIBRARIES}
)

include_directories(
  ${MPI_CXX_INCLUDE_PATH}
)

install(TARGETS ${CODE} DESTINATION .)


