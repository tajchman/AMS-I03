cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

include_directories(
   ${CMAKE_INSTALL_PREFIX}/include
)
find_library (UTIL_LIB Util
  ${CMAKE_INSTALL_PREFIX}/lib)

add_executable(Ex2
  main.cpp
  )


target_link_libraries(Ex2
  ${UTIL_LIB}
  )

install(TARGETS Ex2 DESTINATION .)
