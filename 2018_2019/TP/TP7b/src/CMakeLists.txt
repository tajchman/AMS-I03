cmake_minimum_required( VERSION 2.8 )

enable_language(CXX CUDA)

add_subdirectory(perfs)
add_subdirectory(cpu)
add_subdirectory(gpu)


