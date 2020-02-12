#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#define NTHREADS omp_get_num_threads()
#define ITHREAD  omp_get_thread_num()
#else
#define NTHREADS 1
#define ITHREAD  0
#endif

#include "charge.hxx"
#include "sin.hxx"
#include "timer.hxx"

void init(std::vector<double> & pos,
          std::vector<double> & v1,
          std::vector<double> & v2,
          int n1, int n2)
{
  double pi = 3.14159265;
  int i, n = pos.size();
  
  for (i=n1; i<n2; i++) {
    pos[i] = i*2*pi/n;
    v1[i] = sinus_machine(pos[i]);
    v2[i] = sinus_taylor(pos[i]);
  }
}

void save(const char *filename,
	  std::vector<double> & pos,
	  std::vector<double> & v1,
	  std::vector<double> & v2)
{
  std::ofstream f(filename);

  f  << "# x sin(systeme) approximation" << std::endl;
  int i, n = pos.size();
  for (i=0; i<n; i++)
    f << pos[i] << " " << v1[i] << " " << v2[i] << std::endl;
}

void stat(const std::vector<double> & v1,
          const std::vector<double> & v2,
          int n1, int n2,
          double & sum1, double & sum2)
{
  double s1 = 0.0, s2 = 0.0, err;
  int i;
  for (i=n1; i<n2; i++) {
    err = v1[i] - v2[i];
    s1 += err;
    s2 += err*err;
  }

#pragma omp atomic
  sum1 += s1;

#pragma omp atomic
  sum2 += s2;
}

int main(int argc, char **argv)
{
  Timer T_total;
  T_total.start();
  
  int nthreads;
  #pragma omp parallel
  {
    #pragma omp master
    nthreads = NTHREADS;
  }

  size_t n = argc > 1 ? strtol(argv[1], nullptr, 10) : 2000;
  int imax = argc > 2 ? strtol(argv[2], nullptr, 10) : IMAX;
  set_terms(imax);

  std::cout << "\n\nversion OpenMP grossier 2 : \n\t" << nthreads << " thread(s)\n"
            << "\ttaille vecteur = " << n << "\n"
            << "\ttermes (formule Taylor) : " << imax
            << std::endl;

  Charge C(n, nthreads);
  
  std::vector<double> pos(n), v1(n), v2(n);
  double m, e;

  m = 0;
  e = 0;

  std::vector<double> elapsed_init(nthreads), elapsed_stat(nthreads);
  
#pragma omp parallel shared(pos, v1, v2, n, C)
  {
    Timer T_init, T_stat;
    int ithread = ITHREAD;
    int n1 = C.min(ithread);
    int n2 = C.max(ithread);
    
    T_init.start();
    init(pos, v1, v2, n1, n2);
    T_init.stop();
    elapsed_init[ithread] =  T_init.elapsed();;

    #pragma omp single 
    if (n < 10000)
      save("sinus.dat", pos, v1, v2);
 
    T_stat.start();
    stat(v1, v2, n1, n2, m, e);
    T_stat.stop();
    elapsed_stat[ithread] =  T_stat.elapsed();
  }

  m = m/n;
  e = sqrt(e/n - m*m);

  C.update(elapsed_init);
  
  std::cout << "erreur moyenne : " << m << " ecart-type : " << e
            << std::endl << std::endl;
  
  for (int i=0; i<nthreads; i++)
    std::cout << "time (thread " << i << ") : init " << elapsed_init[i]
              << "s stat " << elapsed_stat[i] << std::endl;

  T_total.stop();
  std::cout << "time : " << T_total.elapsed() << std::endl;
  return 0;
}
