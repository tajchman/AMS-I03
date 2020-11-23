Exemple OpenMP 3 : Addition de vecteurs
_____________________________________________________________________
Pour compiler:

  Se mettre dans le répertoire qui contient ce fichier
  Taper:

    g++ -fopenmp main.cxx -o ex_2_openmp3.exe

  Si tout s'est bien passé : un fichier ex_2_openmp3.exe est créé dans le 
  répertoire

_____________________________________________________________________
Pour exécuter:

  Taper :

    OMP_NUM_THREADS=3 time ./ex_2_openmp3.exe

    
  
