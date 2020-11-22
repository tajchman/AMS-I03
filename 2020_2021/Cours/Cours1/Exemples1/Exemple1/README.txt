Exemple 1 : mesure fine d'une instruction (avec la librairie PAPI)

_____________________________________________________________________
Attention:

  Pour que l'exemple fonctionne, il faut que la commande suivant :
     
     cat /proc/sys/kernel/perf_event_paranoid
     
  affiche 0
  
  Sinon, si vous avec les droits administrateur (root) sur la machine, tapez

    sudo bash
    echo 0 > /proc/sys/kernel/perf_event_paranoid
    exit
_____________________________________________________________________
Pour compiler:

  Se mettre dans le répertoire qui contient de ce fichier
  Taper:

    cmake -DMESURE=ON -DCMAKE_BUILD_TYPE=Release .
    make 

  Si tout s'est bien passé : un fichier ex_1_1.exe est créé dans le 
  répertoire

_____________________________________________________________________
Pour exécuter:

  Taper :

    ./ex_1_1.exe

    Le calcul fait une boucle sur 8 itérations.
    La mesure du temps calcul pour chaque itération est dans un graphe dans le
    fichier cycles.pdf (nombres de cycles processeurs pour chque itération)

    Affichier le fichier cycles.pdf
    
  
