

add_subdirectory(sequential)

find_package(OpenMP)
if(OPENMP_FOUND)
  add_subdirectory(openmp)
else()
  message(STATUS "Open not available : OpenMP versionwill not be compiled")
endif()



