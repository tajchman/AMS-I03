project(SinusSeq)
cmake_minimum_required(VERSION 3.08)

set(CMAKE_RULE_MESSAGES OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(code "sinus_seq.exe")

add_executable(${code} main.cxx sin.cxx calcul.cxx stat.cxx save.cxx)



