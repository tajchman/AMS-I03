cmake_minimum_required( VERSION 2.8 )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")

find_package(KOKKOS)

if (KOKKOS_ENABLE_OPENMP)

include(version_OpenMP.cmake)

elseif (KOKKOS_ENABLE_CUDA)

include(version_Cuda.cmake)

endif()

