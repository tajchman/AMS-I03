set(CODE image_cpu_openmp)

set(CMAKE_C_FLAGS 
     "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS 
     "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS 
     "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

include_directories(
  ${CMAKE_SOURCE_DIR}/common
  ${CMAKE_SOURCE_DIR}/util
  )

add_executable(${CODE}
  ${CMAKE_SOURCE_DIR}/common/main.cxx
  process.cxx
  )

target_link_libraries(${CODE}
  common
  ${PNG_LIBRARIES}
  )

install(TARGETS ${CODE} DESTINATION .)

