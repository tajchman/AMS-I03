cmake_minimum_required(VERSION 2.8)

project(TBB)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")

find_package(TBB)

include_directories(
  ${TBB_INCLUDE_DIRS}
  )


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

add_executable(
  fib_seq.exe 
  main_seq.cxx
  arguments.cxx)

install(TARGETS fib_seq.exe DESTINATION .)
 
set(v tasks)
add_executable(
  fib_${v}.exe 
  main_${v}.cxx
  arguments.cxx)
  
target_compile_options(fib_${v}.exe PUBLIC ${OpenMP_CXX_FLAGS})
target_link_libraries(fib_${v}.exe ${OpenMP_CXX_LIBRARIES})
install(TARGETS fib_${v}.exe DESTINATION .)


