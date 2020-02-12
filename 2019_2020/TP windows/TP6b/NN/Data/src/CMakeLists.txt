cmake_minimum_required(VERSION 3.2)
project(View)

find_package(X11)

include_directories(${X11_INCLUDE_DIR})
add_executable(ViewMNIST ViewMNIST.cxx)
target_link_libraries(ViewMNIST ${X11_LIBRARIES})

install(TARGETS ViewMNIST)
