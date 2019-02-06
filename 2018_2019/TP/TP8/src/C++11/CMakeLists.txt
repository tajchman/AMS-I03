cmake_minimum_required( VERSION 2.8 )

foreach (i 1 2 3 4 5 6)
  add_executable(Test${i} 
    main${i}.cxx 
    )
  
  install(TARGETS Test${i} DESTINATION .)
endforeach()
