set(CODE image_gpu_cuda)

find_package(OpenCL)

if (OpenCL_FOUND)

  include_directories(
    ${OpenCL_INCLUDE_DIRS}
  )

  add_definitions(-DCL_TARGET_OPENCL_VERSION=${OpenCL_VERSION})
  add_definitions(-DINSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")

  set(OpenCL_VERSION ${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0)
  
  include_directories(
    ${OpenCL_INCLUDE_DIRS}
    )

  add_executable(image_gpu_opencl 
    ${CMAKE_SOURCE_DIR}/main.cxx
    process.cxx
    OpenCL.cxx
    )
  
  target_link_libraries(image_gpu_opencl
    util
    ${PNG_LIBRARIES}
    ${OpenCL_LIBRARIES}
    )

  install(TARGETS image_gpu_opencl DESTINATION .)
  install(FILES gray.cl smooth.cl sobel.cl DESTINATION .)
endif()

