set(CODE HeatKokkos_Cuda)

message(STATUS "CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")
set(CMAKE_CXX_FLAGS 
    "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(${CODE} main.cxx )

include_directories(
  ${KOKKOS_INCLUDE_DIR}
)

target_link_libraries(${CODE}
  ${KOKKOS_LIBRARY}
  dl
)

install(TARGETS ${CODE} DESTINATION .)
