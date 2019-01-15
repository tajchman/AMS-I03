find_package(OpenMP)

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
  ${CMAKE_SOURCE_DIR}/common
  )

add_executable(image_cpu_openmp
  main.cxx
  process.cxx
  )

target_link_libraries(image_cpu_openmp
  common
  ${PNG_LIBRARIES}
  )

install(TARGETS image_cpu_openmp DESTINATION .)
