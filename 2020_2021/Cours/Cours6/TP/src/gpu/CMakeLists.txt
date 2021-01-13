
add_executable(inverse_gpu main.cxx matrice.cu)

install(TARGETS inverse_gpu DESTINATION .)
