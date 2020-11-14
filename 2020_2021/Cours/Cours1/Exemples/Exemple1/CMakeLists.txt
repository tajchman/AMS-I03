project(Ex1)
cmake_minimum_required(VERSION 3.08)

set(CMAKE_RULE_MESSAGES OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")

find_package(PAPI REQUIRED)

include_directories(
    ${PAPI_INCLUDE_DIRS}
)

add_compile_options(-Wno-unused-result)
if (MESURE)
   add_compile_options(-DMESURE)
endif()

add_executable(ex1.exe main.cxx init.cxx calcul.cxx timer_papi.cxx)
target_link_libraries(ex1.exe ${PAPI_LIBRARIES})
