set(code "sinus_std_threads")
  
foreach (i v1 v2)
  add_executable(${code}_${i} sinus_${i}.cxx ../sin.cxx)
  target_link_libraries(${code}_${i} util pthread)
  install(TARGETS ${code}_${i} DESTINATION .)
endforeach()
