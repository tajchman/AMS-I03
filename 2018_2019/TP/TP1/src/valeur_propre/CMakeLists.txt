
include_directories(../matrice)
include_directories(../util)

foreach(i 1 2 3 4)
  add_executable(power${i} power${i}.cxx)
  target_link_libraries(power${i} linalg util)
  install(TARGETS power${i} DESTINATION .)
endforeach()
                    
foreach(i 1 2)
  add_executable(power${i}f power${i}.f90)
  target_link_libraries(power${i}f util)
  install(TARGETS power${i}f DESTINATION .)
endforeach()
