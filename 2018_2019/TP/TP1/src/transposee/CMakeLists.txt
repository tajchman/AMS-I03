
include_directories(../matrice)

foreach(i 1 2 3 4 5)
  add_executable(transpose${i} transpose${i}.cpp)
  target_link_libraries(transpose${i} linalg)
endforeach()
