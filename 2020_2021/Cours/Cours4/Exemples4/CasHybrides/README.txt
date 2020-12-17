Exemple de combinaison MPI - OpenMP

2 processus MPI chacuns avec 2 threads

Le processus 0 envoie un message vers le processus 1 
(partie mono-thread)

Puis chacun des threads du processus 0 envoient un message à 
un des threads du processus 1

2 versions: avec MPI single et MPI funneled
_______________________________
Pour compiler:

./build.py
_______________________________
Pour executer:

Si vous avez une connexion graphique:

mpirun -n 2 -xterm -1! ./install/thread_single.exe 
mpirun -n 2 -xterm -1! ./install/thread_multiple.exe 

Sinon:

mpirun -n 2 ./install/thread_single.exe 
mpirun -n 2 ./install/thread_multiple.exe 
