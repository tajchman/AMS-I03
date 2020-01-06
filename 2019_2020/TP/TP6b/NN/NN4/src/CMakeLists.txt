cmake_minimum_required(VERSION 3.14)
set(CMAKE_INSTALL_MESSAGE LAZY)
  
add_compile_definitions(BOOST_UBLAS_NDEBUG)
add_executable(NeuralNet NeuralNet.cxx)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(NeuralNet PUBLIC OpenMP::OpenMP_CXX)
endif()

install(TARGETS NeuralNet DESTINATION .)
