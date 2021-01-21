get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

find_package(OpenCL)

if (OpenCL_FOUND)

  include_directories(
    ${OpenCL_INCLUDE_DIRS}
  )
  
  add_executable(${CODE}
    main.cxx
    calcul_OpenCL.cxx
    ${CMAKE_SOURCE_DIR}/util/timer.cxx
    )

  install(TARGETS ${CODE} DESTINATION .)

endif()
