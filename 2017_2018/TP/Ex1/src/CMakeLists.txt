project (Ex1 C CXX)
  
cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
  ${CMAKE_SOURCE_DIR}/cmake_modules 
)

set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Ex1
  util.c
  main.c
  )

add_executable(Ex1_0
  util.c
  main0.c
  )

