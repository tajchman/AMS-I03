#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <omp.h>

#include "calcul.hxx"
#include "stat.hxx"
#include "sin.hxx"
#include "save.hxx"
#include "timer.hxx"
#include "arguments.hxx"

int main(int argc, char **argv)
{
  Timer T_total, T_calcul, T_stat, T_save;
  T_total.start();
  
  {
    Arguments A(argc, argv);
    size_t n = A.Get("n", 2000);
    size_t imax = A.Get("imax", IMAX);
    int nThreads = A.Get("threads", omp_get_max_threads());

    std::cout << "\nn       : " << n
	      << "\nthreads : " << nThreads
	      << "\n" << std::endl;

    set_terms(imax);

    std::vector<double> pos(n, 0), v1(n, 0), v2(n, 0);

    std::vector<double> elapsed(nThreads);
    
    T_calcul.start();

    omp_set_num_threads(nThreads);

    #pragma omp parallel default(shared)
    {
      Timer T_thread;
      T_thread.start();

      int nThreads = omp_get_num_threads();
      int iThread = omp_get_thread_num();
      int dn = n/nThreads;

      int n1 = iThread * dn;
      int n2 = n1 + dn;
      if (iThread == nThreads-1)
        n2 = n;
      
      calcul(pos, v1, v2, n1, n2);

      T_thread.stop();
      elapsed[iThread] = T_thread.elapsed();
    }
 
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
