Exemple 2.2 : Hello World

_____________________________________________________________________
Pour compiler:

  Se mettre dans le répertoire qui contient ce fichier
  Taper:

    g++ -fopenmp main.cxx -o ex_2_2.exe

  Si tout s'est bien passé : un fichier ex_2_2.exe est créé dans le 
  répertoire

_____________________________________________________________________
Pour exécuter:

  Taper :

    export OMP_NUM_THREADS=3
    ./ex_2_2.exe

    
  
