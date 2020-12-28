Exemple 5 : Diminuer le nombre de tests

_____________________________________________________________________
Pour compiler:

  Se mettre dans le répertoire qui contient de ce fichier
  Taper:

    cmake -DCMAKE_BUILD_TYPE=Release .
    mak

  Si tout s'est bien passé : un fichier ex_1_5.exe est créé dans le 
  répertoire

_____________________________________________________________________
Pour exécuter:

  Taper :

    ./ex_1_5.exe n test1 test2

    où n est un entier positif (taille des vecteur)
    si n n'est pas spécifié, le code prend n = 10000000
    
    test1 et test2 sont égaux soit à 0 soit à 1
 
    On calcule l'expression :

      si test1 == 1:
        u = u + a * v
      si test2 == 1:
        u = u + b * w

    où u, v, w sont des vecteurs de taille n, a et b sont des scalaires
    
    On compare les temps calcul de 2 versions: 
       si on fait le test à l'intérieur de la boucle
       si on fait le test à l'extérieur de la boucle
    
  
