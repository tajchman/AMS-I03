project(SinusAdaptatif)
cmake_minimum_required(VERSION 3.08)

set(CMAKE_RULE_MESSAGES OFF)
#set(CMAKE_VERBOSE_MAKEFILE ON)

set(code "sinus_omp_adaptatif.exe")

find_package(OpenMP REQUIRED)

add_executable(${code} main.cxx arguments.cxx sin.cxx calcul.cxx stat.cxx save.cxx charge.cxx)

target_compile_options(${code} PUBLIC ${OpenMP_CXX_FLAGS})
target_link_libraries(${code} ${OpenMP_CXX_LIBRARIES})

install(TARGETS ${code} DESTINATION .)


