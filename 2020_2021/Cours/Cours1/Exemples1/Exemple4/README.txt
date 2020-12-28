Exemple 4 : Déroulement de boucle

_____________________________________________________________________
Pour compiler:

  Se mettre dans le répertoire qui contient de ce fichier
  Taper:

    cmake -DCMAKE_BUILD_TYPE=Debug .
    make

  Si tout s'est bien passé : un fichier ex_1_4.exe est créé dans le 
  répertoire

_____________________________________________________________________
Pour exécuter:

  Taper :

    ./ex_1_4.exe n

    où n est un entier positif (taille des vecteur)
    si n n'est pas spécifié, le code prend n = 10000000

    On calcule l'expression :

      y = a * x + b

    où x et y sont des vecteurs de taille n, a et b sont des scalaires
    
    On compare les temps calcul de 2 versions: 
       une boucle sur n itérations (chaque itération calcule 1 composante de y)
       une boucle sur n/4 itérations (chaque itération calcule 4 composantes)
    
  
