get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

find_package(CUDA)

if (CUDA_FOUND)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 37 50 75)

  include_directories(
    reduction
    ${CUDA_TOOLKIT_INCLUDE}
    )

  add_executable(${CODE}
    main.cxx
    Calcul_Cuda.cu
   reduction/reduction_kernel.cu
    ${CMAKE_SOURCE_DIR}/util/timer.cxx
    )

  set_source_files_properties(Calcul_Cuda.cu 
    PROPERTIES COMPILE_FLAGS -Wno-deprecated-gpu-targets)

  install(TARGETS ${CODE} DESTINATION .)

endif()

