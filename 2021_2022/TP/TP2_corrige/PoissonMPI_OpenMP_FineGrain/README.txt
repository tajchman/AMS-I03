Code de resolution approchée de l'équation de la chaleur

Version MPI - OpenMP Fine Grain

_____________________
Pour compiler

./build.py

_____________________
Pour exécuter sur 3 processus MPI, chacun avec 2 threads

mpirun -n 3 ./install/Release/PoissonMPI_FineGrain threads=2

ou :

python run.py -n 3 -t 2

ou (sur gin.ensta.fr) :

python submit_hybrid.py -n 3 -t 2
