
add_executable(inverse_cpu main.cxx matrice.cxx timer.cxx)

install(TARGETS inverse_cpu DESTINATION .)
