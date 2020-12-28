if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

add_executable(compteMT_4
  main_4.c 
  count_2.c
  random_2.c
  )


