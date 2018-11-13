cmake_minimum_required(VERSION 3.0)
project(TP1 CXX Fortran)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_subdirectory(timer)
add_subdirectory(matrice)
add_subdirectory(transposee)
add_subdirectory(valeur_propre)


