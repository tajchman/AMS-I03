#include "init.hxx"
#include "calcul.hxx"
#include <fstream>
#include <vector>
#include <cstdlib>

int main(int argc, char **argv) {

   size_t n = argc > 1 ? strtol(argv[1], nullptr, 10) : 10;
   std::vector<double> u(n), v(n);

   init(u);

   calcul(v, u);

#if MESURE
   std::ofstream f("results.gnp");
   f << "set term pdf\n"
        "set output 'cycles.pdf'\n"
        "set xlabel 'Itération'\n"
        "set ylabel 'Cycles'\n"
        "set title 'Nombre de cycles pour une iteration'\n"
        "plot 'results.dat' w lp lc 6 lw 3 pt 6 ps 0.5 notitle\n";
   f.close();
   (void) system("gnuplot results.gnp");
#endif

   return 0;
 }
