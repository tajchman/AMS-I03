project(Ex1)
cmake_minimum_required(VERSION 3.08)
set(CMAKE_RULE_MESSAGES OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

add_executable(ex_seq_2_1.exe 
                main_seq.cxx 
                init.cxx 
                calcul_seq.cxx)

find_package(OpenMP REQUIRED)

add_executable(ex_par_2_1.exe 
                main_par.cxx 
                init.cxx 
                calcul_seq.cxx
                calcul_par0.cxx
                calcul_par1.cxx
                calcul_par2.cxx
                calcul_par3.cxx)

target_compile_options(ex_par_2_1.exe PUBLIC ${OpenMP_CXX_FLAGS})
target_link_libraries(ex_par_2_1.exe ${OpenMP_CXX_LIBRARIES})
                
