set(VERSION cpu_openmp)
set(CODE image_${VERSION})

configure_file(../../version.hxx.in version.hxx @ONLY)


set(CMAKE_C_FLAGS 
     "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS 
     "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS 
     "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND
    CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /debug:full")
  endif()  
endif()

add_executable(${CODE}
  ${CMAKE_SOURCE_DIR}/main.cxx
  process.cxx
  )

target_link_libraries(${CODE}
  util
  ${PNG_LIBRARIES}
  )

install(TARGETS ${CODE} DESTINATION .)

