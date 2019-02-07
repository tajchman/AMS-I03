cmake_minimum_required( VERSION 2.8 )

include_directories(
  ${KOKKOS_INCLUDE_DIR}
)

add_executable(HeatKokkos 
    main.cxx 
)

install(TARGETS HeatKokkos DESTINATION .)
