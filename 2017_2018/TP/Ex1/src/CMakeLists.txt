cmake_minimum_required(VERSION 3.0)

include_directories(
   ${CMAKE_SOURCE_DIR}
   ${CMAKE_INSTALL_PREFIX}/include
)
find_library (UTIL_LIB Util
  ${CMAKE_INSTALL_PREFIX}/lib)

add_executable(Ex1
  main.c
  )

add_executable(Ex1_0
  main0.c
  )

target_link_libraries(Ex1
  ${UTIL_LIB}
  )

target_link_libraries(Ex1_0
  ${UTIL_LIB}
  )

install(TARGETS Ex1 Ex1_0 DESTINATION .)
