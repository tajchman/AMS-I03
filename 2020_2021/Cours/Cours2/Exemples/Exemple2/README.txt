Exemple 2.2 : cout du contrôle de cache

_____________________________________________________________________
Pour compiler:

  Se mettre dans le répertoire qui contient de ce fichier
  Taper:

    cmake -B build -DCMAKE_BUILD_TYPE=Release .
    make -C build

  Si tout s'est bien passé : un fichier ex_2_2.exe est créé dans le 
  répertoire build

_____________________________________________________________________
Pour exécuter:

  Taper :

    ./build/ex_2_2.exe threads offset

    où threads est un entier positif (nombre de threads)
    si threads n'est pas spécifié, le code prend threads = 1 (calcul
    sequentiel)

    et offset est le décalage en espace entre les composantes du 
    vecteur résultat, si offset n'est pas spécifié, offset=1
    
    On compare les temps calcul de 3 versions 
    (sequentielle, 
    parallele sur n threads avec offset=1,
    parallele sur n threads avec offset > 1)
    
  
