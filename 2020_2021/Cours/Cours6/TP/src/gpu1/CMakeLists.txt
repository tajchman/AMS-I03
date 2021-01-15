
add_executable(inverse_gpu1 main.cxx matrice.cu timer.cxx)

install(TARGETS inverse_gpu1 DESTINATION .)
