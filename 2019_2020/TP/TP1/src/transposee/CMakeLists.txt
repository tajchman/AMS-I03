
include_directories(../matrice)
include_directories(../util)

foreach(i 1 2 3 4)
  add_executable(transpose${i} transpose${i}.cxx)
  target_link_libraries(transpose${i} linalg util)
  install(TARGETS transpose${i} DESTINATION .)
endforeach()
