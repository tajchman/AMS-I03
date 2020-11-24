#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "timer.hxx"
#include "GetPot.hxx"
#include "somme.hxx"
#include <omp.h>

int main(int argc, char **argv)
{
  GetPot G(argc, argv);
  int n = G("n", 5000);
  int threads = G("threads", 4);
  int cutoff = G("cutoff", 100);

  set_cutoff(cutoff);
  omp_set_num_threads(threads);

  std::vector<double> v(n);

  int i;
  for (i=0; i<n; i++)
    v[i] = sin(i*M_PI/n);

  Timer T1, T2;

  T1.start();
  double r1 = somme_seq(v);
  T1.stop();
 
  T2.start();
  double r2 = somme_par(v);
  T2.stop();

  std::cout << "calcul seq : " << r1 << "\ttemps CPU " << T1.elapsed() << " s" << std::endl;
  std::cout << "calcul par : " << r2 << "\ttemps CPU " << T2.elapsed() << " s" << std::endl;
  std::cout << "speedup " << T1.elapsed()/T2.elapsed() << std::endl;
  return 0;
}
