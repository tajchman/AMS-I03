find_package(OpenCL)

if (OpenCL_FOUND)
  include_directories(
    ${OpenCL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/util
  )

  add_executable(image_opencl 
    ../../common/main.cxx
    process.cxx
    OpenCL.cxx
  )

  install(TARGETS image_opencl DESTINATION .)

  target_link_libraries(image_opencl
    common
    ${PNG_LIBRARIES}
  )

else()
  message(STATUS "OpenCL not found : cannot compile OpenCL version")
endif()
