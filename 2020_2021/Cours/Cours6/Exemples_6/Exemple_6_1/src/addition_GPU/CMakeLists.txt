get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(CMAKE_CUDA_ARCHITECTURES OFF)

add_executable(${CODE}
  main.cxx
  calcul_GPU.cu
  timerGPU.cu)


install(TARGETS ${CODE} DESTINATION .)
