get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

find_package(CUDA)

if (CUDA_FOUND)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 37 50 75)

  include_directories(
    ${CUDA_TOOLKIT_INCLUDE}
    )

  add_executable(${CODE}
    main.cxx
    Calcul_Cuda.cu
    ${CMAKE_SOURCE_DIR}/util/timer.cxx
    )

  install(TARGETS ${CODE} DESTINATION .)

endif()

