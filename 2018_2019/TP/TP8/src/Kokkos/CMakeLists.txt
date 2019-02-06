cmake_minimum_required( VERSION 2.8 )

add_executable(HeatKokkos 
    main.cxx 
)

install(TARGETS HeatKokkos DESTINATION .)
