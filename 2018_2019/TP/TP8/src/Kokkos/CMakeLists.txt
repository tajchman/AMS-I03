cmake_minimum_required( VERSION 2.8 )
enable_language(CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")

find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)
find_package(KOKKOS REQUIRED)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories(
  ${KOKKOS_INCLUDE_DIR}
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")

add_executable(HeatKokkos 
    main.cxx 
)
target_link_libraries(HeatKokkos
  ${KOKKOS_LIBRARY}
  )

install(TARGETS HeatKokkos DESTINATION .)
