
include_directories(../matrice)
include_directories(../timer)

foreach(i 1 2 3 4 5 6)
  add_executable(power${i} power${i}.cpp ../matrice/Util.c)
  target_link_libraries(power${i} linalg timer)
  install(TARGETS power${i} DESTINATION .)
endforeach()
                    
foreach(i 1 2)
  add_executable(power${i}f power${i}.f90 ../matrice/Util.c)
  install(TARGETS power${i}f DESTINATION .)
endforeach()
