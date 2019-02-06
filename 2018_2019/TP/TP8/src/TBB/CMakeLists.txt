cmake_minimum_required( VERSION 2.8 )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")

find_package(TBB)

include_directories(
  ${TBB_INCLUDE_DIRS}
  )

foreach(d 0 1)
  add_executable(heatTBB_${d} main.cxx heatTBB_${d}.cxx save.cxx)
  target_link_libraries(heatTBB_${d} ${TBB_LIBRARIES})
  install(TARGETS heatTBB_${d} DESTINATION .)
endforeach()


