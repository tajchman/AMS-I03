get_filename_component(CODE ${CMAKE_CURRENT_SOURCE_DIR} NAME)

find_package(OpenCL)

if (OpenCL_FOUND)

  include_directories(
    ${OpenCL_INCLUDE_DIRS}
    reduction
  )

  add_definitions(-DCL_TARGET_OPENCL_VERSION=${OpenCL_VERSION})
  add_definitions(-DINSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")

  add_executable(${CODE}
    main.cxx
    Calcul_OpenCL.cxx
    OpenCL.cxx
    ${CMAKE_SOURCE_DIR}/util/timer.cxx
    reduction/reduce.cxx
    )

  target_link_libraries(${CODE} ${OpenCL_LIBRARIES})
  install(TARGETS ${CODE} DESTINATION .)
  install(FILES init.cl add.cl reduction/reduce.cl DESTINATION .)
endif()
