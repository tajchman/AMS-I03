cmake_minimum_required( VERSION 2.8 )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")

find_package(TBB)

include_directories(
  ${TBB_INCLUDE_DIRS}
  )

add_executable(heatTBB main.cxx heatTBB.cxx save.cxx)

install(TARGETS heatTBB DESTINATION .)
