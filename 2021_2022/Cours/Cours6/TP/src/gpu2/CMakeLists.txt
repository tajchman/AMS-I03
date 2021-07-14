
add_executable(inverse_gpu2 main.cxx matrice.cu timer.cxx)

install(TARGETS inverse_gpu2 DESTINATION .)
