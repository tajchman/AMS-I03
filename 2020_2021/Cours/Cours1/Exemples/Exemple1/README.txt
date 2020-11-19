Exemple 1 : mesure fine d'une instruction (avec la librairie PAPI)

Pour compiler:

  Se mettre dans le répertoire qui contient de ce fichier
  Taper:

    mkdir -p build
    cd build
    cmake -DMESURE=ON .
    cd ..

  Si tout s'est bien passé : un fichier ex1.exe est créé

Pour exécuter:

  Taper :

    ./build/ex1.exe

    Le calcul fait une boucle
  