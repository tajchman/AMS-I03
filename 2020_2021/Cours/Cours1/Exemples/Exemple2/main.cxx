#include "init.hxx"
#include "calcul.hxx"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "timer.hxx"

#define SIZE 100000000
int main(int argc, char **argv) {

   std::ofstream f("results.dat");

   std::cerr << "distance " << std::endl;

   std::vector<double> u(SIZE, 0);
   int k;
   for (k=1; k<=4096; k *= 2) {

       std::cerr << " " << k;
       Timer T;
       T.start();
      
       calcul(u, k);
    
       T.stop();
       double Trel = T.elapsed() / SIZE * k;
       f << k << " " << Trel << std::endl;
       std::cerr << " " << Trel << std::endl;
   }

   f.close();
   std::ofstream r("results.gnp");
   r << "set term pdf\n"
        "set output 'temps_cache.pdf'\n"
        "set xlabel 'Décalage'\n"
        "set ylabel 'Temps CPU/nombres acces'\n"
        "set logscale x\n"
        "plot 'results.dat' w lp lc 6 lw 3 pt 6 ps 0.5 notitle\n";
   r.close();
   (void) system("gnuplot results.gnp");

   return 0;
 }
