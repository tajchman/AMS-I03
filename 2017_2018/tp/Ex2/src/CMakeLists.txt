project (Ex2 C CXX)

cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${CMAKE_SOURCE_DIR}
)

set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PAPI)

include_directories(
  ${PAPI_INCLUDE}
  )

add_executable(Ex2
  util.cpp
  main.cpp
  )

target_link_libraries(
  Ex2
  ${PAPI_LIBRARIES}
  )
