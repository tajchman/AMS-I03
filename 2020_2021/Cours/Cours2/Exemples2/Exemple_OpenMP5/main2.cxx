#include <iostream>
#include <vector>
#include <cstdlib>
#include <time.h>
#include "timer.hxx"
#ifdef _OPENMP
#include <omp.h>
#endif

#define offset 20

int main(int argc, char **argv) {

  Timer T_total, T_init, T_calcul;
  T_total.start();

  double moy, var;

  {
  T_init.start();
  
  int i, n = argc > 1 ? strtol(argv[1], NULL, 10) : 100000000;
  std::vector<double> u(n);

  srand (time(NULL));
  srand (0);
  for (i=0; i<n; i++)
    u[i] = 0.5 + double(rand())/RAND_MAX;

  int nThreads;

  #ifdef _OPENMP
  #pragma omp parallel
  {
    #pragma omp master
    nThreads = omp_get_num_threads();
  }
  #else
    nThreads = 1;
  #endif

  T_init.stop();
  T_calcul.start();

  double s, s2;
  std::vector<double> s_partiel(offset*nThreads), s2_partiel(offset*nThreads);
  int iTh;

  #pragma omp parallel for default (shared) private (iTh)
  for (i = 0; i < n ; i++) {

#ifdef _OPENMP
    iTh = omp_get_thread_num() * offset;
#else
    iTh = 0;
#endif
    s_partiel[iTh] += u[i];
    s2_partiel[iTh] += u[i]*u[i];
  }

  s = 0.0; s2 = 0.0;
  for (iTh=0; iTh < offset*nThreads; iTh+=offset) {
    s += s_partiel [iTh];
    s2 += s2_partiel [iTh];
  }
  moy = s / n;
  var = s2/n - moy*moy;

  T_calcul.stop();

  }
  T_total.stop();

  std::cout << "moyenne : " << moy << " variance " << var << std::endl;
  std::cout << "temps init   CPU : " << T_init.elapsed() << " s" << std::endl;
  std::cout << "temps calcul CPU : " << T_calcul.elapsed() << " s" << std::endl;
  std::cout << std::endl;
  std::cout << "temps total  CPU : " << T_total.elapsed() << " s" << std::endl;
  return 0;
}
