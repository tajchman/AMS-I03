get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

include_directories(
  reduction
  )

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 35 50 75)

include_directories(${CUDA_TOOLKIT_INCLUDE})

add_executable(${CODE}
  main.cxx
  calcul_GPU.cu
  reduction/reduction_kernel.cu
  ${CMAKE_SOURCE_DIR}/util/timer.cxx
  )

install(TARGETS ${CODE} DESTINATION .)
