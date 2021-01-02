Code de resolution approchée de l'équation de la chaleur

Version MPI - OpenMP Fine Grain

_____________________
Pour compiler

./build.py

_____________________
Pour exécuter sur 3 processus MPI, chacun avec 2 threads

mpirun -n 3 ./build/Gnu/Release/PoissonMPI_CoarseGrain threads=2
