
find_package(OpenCL REQUIRED)

set(OpenCL_VERSION ${OpenCL_VERSION_MAJOR}.${OpenCL_VERSION_MINOR}.0)
  
include_directories(
  ${OpenCL_INCLUDE_DIRS}
  )

message(STATUS "OpenCL version ${OpenCL_VERSION}")
add_definitions(-DCL_TARGET_OPENCL_VERSION=${OpenCL_VERSION})

add_executable(deviceQuery_opencl
   deviceQuery.cpp
)

target_link_libraries(deviceQuery_opencl ${OpenCL_LIBRARIES})

install(TARGETS deviceQuery_opencl DESTINATION .)


