find_package(OpenCL)

if (OpenCL_FOUND)
  
  set(OpenCL_VERSION ${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0)
  
  include_directories(
    ${OpenCL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/util
    )

  message(STATUS ${OpenCL_VERSION})
  add_definitions(-DCL_TARGET_OPENCL_VERSION=${OpenCL_VERSION})

  add_executable(image_gpu_opencl 
    ../../common/main.cxx
    process.cxx
    OpenCL.cxx
    )
  
  install(TARGETS image_gpu_opencl DESTINATION .)
  
  target_link_libraries(image_gpu_opencl
    common
    ${PNG_LIBRARIES}
    ${OpenCL_LIBRARIES}
    )
  
else()
  message(STATUS "OpenCL not found : cannot compile OpenCL version")
endif()
