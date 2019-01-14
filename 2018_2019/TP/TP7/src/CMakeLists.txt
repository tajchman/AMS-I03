cmake_minimum_required( VERSION 2.8 )

find_package( CUDA REQUIRED )
find_package( OpenMP REQUIRED )

add_subdirectory(common)
add_subdirectory(cpu)
add_subdirectory(cpu_openmp)
add_subdirectory(gpu)
add_subdirectory(exemple)

