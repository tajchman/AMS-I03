cmake_minimum_required(VERSION 2.8)
project(TP3 CXX C)

if(WIN32)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../install
    CACHE PATH "chemin d'installation par defaut" FORCE)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_MESSAGE LAZY)

if (CMAKE_VERSION VERSION_LESS "3.1")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
  else ()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif ()
else ()
  set (CMAKE_CXX_STANDARD 11)
endif ()

find_package(OpenMP)
message(status ${OpenMP_CXX_FLAGS} )
find_package(MPI)

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/util)

add_subdirectory(util)

foreach(d seq openmp_fine openmp_coarse mpi mpi_openmp_fine mpi_openmp_coarse)
   add_subdirectory(sinus_${d})
endforeach()
add_subdirectory(PoissonMPI)

configure_file(run.sh.in run.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/run.sh DESTINATION . 
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)
install(FILES job.in DESTINATION .)



