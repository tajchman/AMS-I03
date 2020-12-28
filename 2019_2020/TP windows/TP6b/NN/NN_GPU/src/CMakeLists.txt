cmake_minimum_required(VERSION 3.14)
project(NN5)
set(CMAKE_INSTALL_MESSAGE LAZY)
  
include_directories(${CMAKE_SOURCE_DIR}/../../communs)

find_package(CUDA)

add_compile_definitions(BOOST_UBLAS_NDEBUG)
cuda_add_executable(NeuralNet NeuralNet.cxx Network.cu MatrixVectorCUDA.cu)

install(TARGETS NeuralNet DESTINATION .)
