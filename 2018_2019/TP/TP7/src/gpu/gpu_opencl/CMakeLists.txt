find_package(OpenCL)

if (OpenCL_FOUND)
  include_directories(
    ${OpenCL_INCLUDE_DIRS}
    ../../common
  )

  add_executable(image_opencl 
    ../../common/main.cxx
    process.cxx
  )

  install(TARGETS image_opencl DESTINATION .)

  target_link_libraries(image_opencl
    common
    ${PNG_LIBRARIES}
  )

else()
  message(STATUS "OpenCL not found : cannot compile OpenCL version")
endif()
