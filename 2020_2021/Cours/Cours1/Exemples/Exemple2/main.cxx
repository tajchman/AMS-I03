#include "init.hxx"
#include "calcul.hxx"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "timer.hxx"

#define SIZE 100000000
int main(int argc, char **argv) {

#ifdef MESURE
   std::ofstream f("results.dat");
#endif
 
   std::vector<double> u(SIZE, 0);
   int k;
   for (k=1; k<=4096; k *= 2) {

#ifdef MESURE
       Timer T;
       T.start();
#endif
      
       calcul(u, k);
    
#ifdef MESURE
       T.stop();
       double Trel = T.elapsed() / SIZE * k;
       f << k << " " << Trel << std::endl;
       std::cerr << k << " " << Trel << std::endl;
#endif
   }
   f.close();

#if MESURE
   std::ofstream r("results.gnp");
   r << "set term pdf\n"
        "set output 'temps_cache.pdf'\n"
        "set xlabel 'Décalage'\n"
        "set ylabel 'Temps CPU/nombres acces'\n"
        "set logscale x\n"
        "plot 'results.dat' w lp lc 6 lw 3 pt 6 ps 0.5 notitle\n";
   r.close();
   (void) system("gnuplot results.gnp");
#endif

   return 0;
 }
