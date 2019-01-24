set(CODE image_gpu_cuda)

set(OpenCL_VERSION ${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0)
  
include_directories(
  ${OpenCL_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/common
  ${CMAKE_SOURCE_DIR}/util
  )

message(STATUS ${OpenCL_VERSION})
add_definitions(-DCL_TARGET_OPENCL_VERSION=${OpenCL_VERSION})
add_definitions(-DINSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")

add_executable(image_gpu_opencl 
  ${CMAKE_SOURCE_DIR}/common/main.cxx
  process.cxx
  OpenCL.cxx
  )
  
target_link_libraries(image_gpu_opencl
  common
  ${PNG_LIBRARIES}
  ${OpenCL_LIBRARIES}
  )

install(TARGETS image_gpu_opencl DESTINATION .)
install(FILES gray.cl smooth.cl sobel.cl DESTINATION .)
