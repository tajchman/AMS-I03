Exemple OpenMP 5 : Moyenne, variance d'un vecteur
_____________________________________________________________________
Pour compiler:

  Se mettre dans le répertoire qui contient ce fichier
  Taper:

    g++ -fopenmp main1.cxx -o ex_2_openmp5_1.exe
    g++ -fopenmp main2.cxx -o ex_2_openmp5_2.exe
    g++ -fopenmp main1.cxx -o ex_2_openmp5_3.exe
    g++ -fopenmp main2.cxx -o ex_2_openmp5_4.exe

  Si tout s'est bien passé : 4 fichiers executables
  ex_2_openmp5_1.exe, ex_2_openmp5_2.exe, ex_2_openmp5_3.exe 
  et ex_4_openmp5_2.exe sont créés dans le répertoire

_____________________________________________________________________
Pour exécuter:

  Taper :

    OMP_NUM_THREADS=1 ./ex_2_openmp5_1.exe
    OMP_NUM_THREADS=3 ./ex_2_openmp5_1.exe

    OMP_NUM_THREADS=1 ./ex_2_openmp5_2.exe
    OMP_NUM_THREADS=3 ./ex_2_openmp5_2.exe

    OMP_NUM_THREADS=1 ./ex_2_openmp5_3.exe
    OMP_NUM_THREADS=3 ./ex_2_openmp5_3.exe
 
    OMP_NUM_THREADS=1 ./ex_2_openmp5_3.exe
    OMP_NUM_THREADS=3 ./ex_2_openmp5_3.exe
   
  
