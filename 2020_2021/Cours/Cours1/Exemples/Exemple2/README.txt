Exemple 2 : Addition de 2 matrices, illustration de la localité spatiale

_____________________________________________________________________
Pour compiler:

  Se mettre dans le répertoire qui contient de ce fichier
  Taper:

    cmake -B build -DCMAKE_BUILD_TYPE=Release .
    make -C build

  Si tout s'est bien passé : un fichier ex2.exe est créé dans le 
  répertoire build

_____________________________________________________________________
Pour exécuter:

  Taper :

    ./build/ex2.exe n

    où n est un entier positif (taille des matrices)
    si n n'est pas spécifié, le code prend n = 1024

    On calcule l'addition de 2 matrices de taille NxM
    en parcourant les matrices de 2 façons: 
    
    - ligne par ligne et dans chaque ligne,
      on parcourt tous les éléments de cette ligne
   
    - colonne par colonne et dans chaque colonne,
      on parcourt tous les éléments de cette colonne
 
    Le temps de calcul des 2 algorithmes est affiché 
    
  
