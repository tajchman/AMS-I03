Exemple 2 : Additions de vecteurs, illustration de la localité temporelle

_____________________________________________________________________
Pour compiler:

  Se mettre dans le répertoire qui contient de ce fichier
  Taper:

    cmake -DCMAKE_BUILD_TYPE=Release .
    make

  Si tout s'est bien passé : un fichier ex_1_3.exe est créé dans le 
  répertoire

_____________________________________________________________________
Pour exécuter:

  Taper :

    ./ex_1_3.exe n

    où n est un entier positif (taille des vecteur)
    si n n'est pas spécifié, le code prend n = 100000

    On calcule 3 additions de vecteurs :

      u = 2a + 3b (1)
      v = 3a + 2b (2)
      w = c + d   (3)
    
    On compare les temps calcul de 2 versions: 
       (1) calculé par une boucle, (2) et (3) par une autre boucle
       (1) et (2) calculés par une boucle, (3) par une autre boucle
    
  
