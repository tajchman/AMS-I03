Exemple 2.2b : Hello World 2

Chaque thread affiche son numéro dans son message
_____________________________________________________________________
Pour compiler:

  Se mettre dans le répertoire qui contient ce fichier
  Taper:

    g++ -fopenmp main.cxx -o ex_2_openmp2.exe

  Si tout s'est bien passé : un fichier ex_2_openmp2.exe est créé dans le 
  répertoire

_____________________________________________________________________
Pour exécuter:

  Taper :

    OMP_NUM_THREADS=3 ./ex_2_openmp2.exe

 
  Remarquer que les affichages sont incorrects. Refaire plusieurs
  exécutions 

_____________________________________________________________________
Deux versions corrigées:

  On propose de remplacer main.cxx par main_corrige.cxx ou
  main_corrige2.cxx

  Examiner les différences avec main.cxx et
  faire les mêmes tests 
    
  
