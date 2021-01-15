
add_executable(inverse_gpu3 main.cxx matrice.cu reduction.cu timer.cxx)

install(TARGETS inverse_gpu3 DESTINATION .)
