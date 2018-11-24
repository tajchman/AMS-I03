set(code "sinus_std_threads")
  
add_executable(${code} sinus.cxx)
add_executable(${code}_2 sinus_2.cxx)
target_link_libraries(${code} util pthread)
target_link_libraries(${code}_2 util pthread)

install(TARGETS ${code} ${code}_2 DESTINATION .)
