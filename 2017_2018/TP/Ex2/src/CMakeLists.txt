cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})
set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(
   ${CMAKE_SOURCE_DIR}
   ${CMAKE_INSTALL_PREFIX}/include
)
find_library (UTIL_LIB Util
  ${CMAKE_INSTALL_PREFIX}/lib)

add_executable(Ex2
  main.cpp
  vecteur.cpp
  matrice.cpp
  )


target_link_libraries(Ex2
  ${UTIL_LIB}
  )

install(TARGETS Ex2 DESTINATION .)
