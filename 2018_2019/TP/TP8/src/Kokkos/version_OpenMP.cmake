
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
set(CMAKE_CXX_FLAGS 
   "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS 
   "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

set(CODE HeatKokkos_OpenMP)
add_executable(${CODE} main.cxx )

include_directories(
  ${KOKKOS_INCLUDE_DIR}
)

target_link_libraries(${CODE}
  ${KOKKOS_LIBRARY}
  dl
)

install(TARGETS ${CODE} DESTINATION .)

