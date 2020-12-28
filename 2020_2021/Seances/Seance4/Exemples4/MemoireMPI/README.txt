Cet exemple minimal appelle seulement MPI\_Init/MPI\_Finalize et mesure la quantité mémoire utilisée par ces appels

Pour compiler:

mpicxx main.cxx memory_used.cxx -o memoirempi.exe

Pour exécuter :

mpirun --host localhost:<n> -n <n> ./memoirempi.exe

où <n> est le nombre de processus mpi
(--host localhost:<n> permet avec OpenMPI de lancer plus de processus MPI qu'il y a de coeurs sur une machine, pour tester seulement)

Tester avec plusieurs valeurs de <n> : 1,2,5,10,20,50,100
L'exécutable affiche, en moyenne, la taille mémoire reservée par MPI_Init et celle libérée par MPI_Finalize

