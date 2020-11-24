#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

#include "calcul.hxx"
#include "stat.hxx"
#include "sin.hxx"
#include "save.hxx"
#include "timer.hxx"
#include "GetPot.hxx"
#include <iostream>
#include <omp.h>

int main(int argc, char **argv)
{
  Timer T_total, T_calcul, T_stat, T_save;
  T_total.start();
  
  {
    GetPot G(argc, argv);
    size_t n = G("n", 2000);
    size_t imax = G("imax", IMAX);
    int nThreads = G("threads", omp_get_max_threads());
    int nTasks = G("tasks", nThreads * 10);
    set_terms(imax);
     
    std::cout << "\nn       : " << n << std::endl;
    std::cout << "threads : " << nThreads << "\n" << std::endl;

    std::vector<double> pos(n, 0), v1(n, 0), v2(n, 0);
   
    T_calcul.start();

    omp_set_num_threads(nThreads);

    int dn = n/nTasks;
    #pragma omp parallel
    {
      #pragma omp master
      {
        for (int i=0; i<nTasks-1; i++)
          #pragma omp task firstprivate(i) 
          {          
            int n_start = i * dn;
            int n_end = (i+1) * dn;
            calcul(pos, v1, v2, n_start, n_end);
            #pragma omp critical
            std::cout << omp_get_thread_num() 
                      << ": n1 = " << n_start 
                      << " n2 = " << n_end << std::endl;
          }

        #pragma omp task
        {
          int n_start = (nTasks-1)*dn;
          int n_end = n;
          calcul(pos, v1, v2, n_start, n_end);
        }
      }
    }
    #pragma omp taskwait

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
      save("sinus_omp_tasks.dat", pos, v1, v2);

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
