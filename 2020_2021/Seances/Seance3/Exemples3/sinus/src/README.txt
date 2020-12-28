On compare ici le calcul de y = sin(x) de deux façons:
- en utilisant la fonction systeme
- en utilisant un développement limté 
  (série de de Taylor)

Ce calcul est fait dans plusieurs répertoires:
- en séquentiel dans sinus_seq
- avec OpenMP grain fin dans 
  sin_omp_fine_grain
- avec OpenMP grain grossier dans 
  sin_omp_coarse_grain
- avec OpenMP grain grossier avec adaptation 
  du découpage dans sinus_omp_adaptatif
- avec des tâches OpenMP dans
  sin_omp_tasks

Pour mieux faire ressortir les différence, le calcul du
développement limité est artificiellement ralenti 
proportionnellement au nombre de termes

______________________________
Pour compiler :

 Se placer dans le répertoire de ce fichier
 et taper:

    python ./build.py

______________________________
Pour exécuter:

 Taper les commandes 

   ./install/sinus_seq.exe 
   ./install/sinus_omp_fine_grain.exe threads=3
   ./install/sinus_omp_coarse_grain.exe threads=3
   ./install/sinus_omp_tasks.exe threads=3
   ./install/sinus_omp_adaptatif.exe threads=3

   (taper plusieurs fois la dernière commande et 
   comparer les temps de calcul)
   
   