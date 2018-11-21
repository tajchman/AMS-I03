set(code "sinus_seq")

add_executable(${code} ${code}.cxx)
target_link_libraries(${code} util)

install(TARGETS ${code} DESTINATION .)
