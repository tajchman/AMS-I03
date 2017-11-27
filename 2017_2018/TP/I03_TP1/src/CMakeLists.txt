cmake_minimum_required(VERSION 3.0)
project(POISSON)

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif()

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
endif()

add_directory(timer)

add_executable(PoissonSeq 
	main.cxx 
	parameters.cxx 
	values.cxx 
	scheme.cxx
	)

add_target_libraries(
   PoissonSeq
   timer
)
