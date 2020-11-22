Exemple 2.1 : Boucle simple, versions séquentielle et parallèle

_____________________________________________________________________
Pour compiler:

  Se mettre dans le répertoire qui contient de ce fichier
  Taper:

    cmake -B build -DCMAKE_BUILD_TYPE=Release .
    make -C build

  Si tout s'est bien passé : un fichier ex_2_1.exe est créé dans le 
  répertoire build

_____________________________________________________________________
Pour exécuter:

  Taper :

    ./build/ex_2_1.exe n

    où n est un entier positif (taille des vecteurs)
    si n n'est pas spécifié, le code prend n = 1000

    On calcule une boucle simple en versions séquentielle et parallèles 
    avec OpenMP (3 versions)
    
    On compare les temps calcul de 3 versions
    
  
