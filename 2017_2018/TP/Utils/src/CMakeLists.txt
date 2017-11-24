cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

include_directories(
   ${CMAKE_SOURCE_DIR}
)

add_library(Util
  util.c
  timer.cpp
  )

install(TARGETS Util DESTINATION lib)
install(FILES util.h DESTINATION include)



