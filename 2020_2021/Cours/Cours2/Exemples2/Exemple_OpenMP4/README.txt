Exemple OpenMP 4 : Addition de matrices
_____________________________________________________________________
Pour compiler:

  Se mettre dans le répertoire qui contient ce fichier
  Taper:

    g++ main1.cxx Matrice.cxx -o ex_2_seq4.exe
    g++ -fopenmp main1.cxx Matrice.cxx -o ex_2_openmp4_1.exe
    g++ -fopenmp main2.cxx Matrice.cxx -o ex_2_openmp4_2.exe
    g++ -fopenmp main3.cxx Matrice.cxx -o ex_2_openmp4_3.exe

  Si tout s'est bien passé : 4 fichiers executables
  ex_2_seq4.exe ex_2_openmp4_1.exe, ex_2_openmp4_2.exe et ex_2_openmp4_3.exe 
  sont créés dans le répertoire

_____________________________________________________________________
Pour exécuter:

  Taper :

    OMP_NUM_THREADS=1 time ./ex_2_openmp4_1.exe
    OMP_NUM_THREADS=1 time ./ex_2_openmp4_2.exe
    OMP_NUM_THREADS=1 time ./ex_2_openmp4_3.exe

    OMP_NUM_THREADS=3 time ./ex_2_openmp4_1.exe
    OMP_NUM_THREADS=3 time ./ex_2_openmp4_2.exe
    OMP_NUM_THREADS=3 time ./ex_2_openmp4_3.exe

    
  
