get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

include_directories(
  reduction
  )

set(CMAKE_CUDA_ARCHITECTURES OFF)

add_executable(${CODE}
  main.cxx
  calcul_GPU.cu
  reduction/reduction_kernel.cu
  timerGPU.cu)


install(TARGETS ${CODE} DESTINATION .)
