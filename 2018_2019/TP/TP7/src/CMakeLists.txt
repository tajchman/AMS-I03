cmake_minimum_required( VERSION 2.8 )

find_package(PNG)

include_directories(
  ${PNG_PNG_INCLUDE_DIR}
)

add_subdirectory(common)
add_subdirectory(cpu)
add_subdirectory(cpu_openmp)
add_subdirectory(gpu)
add_subdirectory(exemples)

