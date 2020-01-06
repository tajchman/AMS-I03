cmake_minimum_required(VERSION 3.14)
project(NN1)
set(CMAKE_INSTALL_MESSAGE LAZY)

include_directories(${CMAKE_SOURCE_DIR}/../../communs)

add_compile_definitions(BOOST_UBLAS_NDEBUG)
add_compile_definitions(BOOST_UBLAS_NDEBUG)
add_executable(NeuralNet NeuralNet.cxx)
install(TARGETS NeuralNet DESTINATION .)
