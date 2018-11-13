
include_directories(../matrice)
include_directories(../timer)

foreach(i 1 2 3)
  add_executable(power${i} power${i}.cpp)
  target_link_libraries(power${i} linalg timer)
endforeach()

foreach(i 1 2)
  add_executable(power${i}f power${i}.f90)
endforeach()
