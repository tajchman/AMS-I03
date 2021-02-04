cmake_minimum_required(VERSION 2.8)

project(Tasks)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_compile_options(-DDEBUG)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qdiag-disable:1786,3180")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=1786,3180")
  endif()
endif()

include_directories(
  ${MPI_INCLUDES}
)

foreach(v thread_single thread_multiple)

  add_executable(
    ${v}.exe
    main_${v}.cxx
  )
  target_compile_options(${v}.exe PUBLIC ${OpenMP_CXX_FLAGS})
  target_link_libraries(${v}.exe ${OpenMP_CXX_LIBRARIES})
  
  install(TARGETS ${v}.exe  DESTINATION .)

endforeach()
