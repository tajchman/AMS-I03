Exemple OpenMP 3 : Addition de vecteurs
_____________________________________________________________________
Pour compiler:

  Se mettre dans le répertoire qui contient ce fichier
  Taper:

    g++ -fopenmp main1.cxx -o ex_2_openmp3_1.exe
    g++ -fopenmp main2.cxx -o ex_2_openmp3_2.exe

  Si tout s'est bien passé : deux fichiers executables
  ex_2_openmp3_1.exe et ex_2_openmp3_2.exe sont créés dans le 
  répertoire

_____________________________________________________________________
Pour exécuter:

  Taper :

    OMP_NUM_THREADS=1 time ./ex_2_openmp3_1.exe
    OMP_NUM_THREADS=1 time ./ex_2_openmp3_2.exe

    OMP_NUM_THREADS=3 time ./ex_2_openmp3_1.exe
    OMP_NUM_THREADS=3 time ./ex_2_openmp3_2.exe

    
  
