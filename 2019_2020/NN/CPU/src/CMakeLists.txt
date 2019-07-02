cmake_minimum_required(VERSION 3.5)

if (SYSTEM_PNG)
  find_package(PNG)
else()
  get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
  get_filename_component(PARENT_DIR ${PARENT_DIR} DIRECTORY)
  set(LIBS_DIR ${CMAKE_SOURCE_DIR} 
  set(LIBS_DIR ${PARENT_DIR}/Libs/install)
  set(PNG_PNG_INCLUDE_DIR ${LIBS_DIR}/include)
  link_directories(${LIBS_DIR}/lib
                   ${LIBS_DIR}/lib64)
  set(PNG_LIBRARIES png z) 
endif()

include_directories(
  ${PNG_PNG_INCLUDE_DIR}
  )

add_executable(nn_CPU main.cxx io_png.cxx reseau.cxx input.cxx)

target_link_libraries(nn_CPU
  ${PNG_LIBRARIES}
  )

install(TARGETS nn_CPU DESTINATION .)

