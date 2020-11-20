project(Ex1)
cmake_minimum_required(VERSION 3.08)
set(CMAKE_RULE_MESSAGES OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(ex1.exe 
                main.cxx 
                init.cxx 
                calcul_seq.cxx
                calcul_par0.cxx
                calcul_par1.cxx
                calcul_par2.cxx)
