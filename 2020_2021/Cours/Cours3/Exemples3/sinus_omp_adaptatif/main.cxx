#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

#include "calcul.hxx"
#include "stat.hxx"
#include "sin.hxx"
#include "save.hxx"
#include "timer.hxx"
#include "charge.hxx"
#include <iostream>
#include <omp.h>

int main(int argc, char **argv)
{
  Timer T_total, T_calcul, T_stat, T_save;
  T_total.start();
  
  {
    size_t n = argc > 1 ? strtol(argv[1], NULL, 10) : 2000;
    int imax = argc > 2 ? strtol(argv[2], NULL, 10) : IMAX;
    set_terms(imax);

    int nThreads;

    #pragma omp parallel
    {
      #pragma omp master
      nThreads = omp_get_num_threads();
    }
    Charge C(n, nThreads); 
    std::vector<double> pos(n, 0), v1(n, 0), v2(n, 0);
    std::vector<double> elapsed(nThreads);

    T_calcul.start();

    #pragma omp parallel default(shared)
    {
      Timer T_thread;
      T_thread.start();

      int iThread = omp_get_thread_num();
      int n1 = C.min(iThread);
      int n2 = C.max(iThread);
      
      calcul(pos, v1, v2, n1, n2);

      T_thread.stop();
      elapsed[iThread] = T_thread.elapsed();
    }
    C.update(elapsed);
 
    T_calcul.stop();
    T_stat.start();

    double m, e;
  
    stat(v1, v2, 0, n, m, e);

    m = m/n;
    e = sqrt(e/n - m*m);
    std::cout << "m = " << m << " e = " << e << std::endl << std::endl;

    T_stat.stop();
    T_save.start();

    if (n <= 5000)
      save("sinus_seq.dat", pos, v1, v2);

    T_save.stop();
  }
  T_total.stop();
  std::cout << "temps calcul : " << T_calcul.elapsed() << " s" << std::endl;
  std::cout << "temps stat   : " << T_stat.elapsed() << " s" << std::endl;
  std::cout << "temps save   : " << T_save.elapsed() << " s" << std::endl;
  std::cout << std::endl;
  std::cout << "temps total  : " << T_total.elapsed() << " s" << std::endl;
  return 0;
}
