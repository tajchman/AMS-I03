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

#include "sin.hxx"

void init(std::vector<double> & pos,
          std::vector<double> & v1,
          std::vector<double> & v2)
{
  double x, pi = 3.14159265;
  int i, n = pos.size();

  v1.resize(n);
  v2.resize(n);

#pragma omp parallel for private(x)
  for (i=0; i<n; i++) {
    x = i*2*pi/n;
    pos[i] = x;
    v1[i] = sinus_machine(x);
    v2[i] = sinus_taylor(x);
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
          double & moyenne, double & ecart_type)
{
  double s1 = 0.0, s2 = 0.0, err;
  int i, n = v1.size();

  #pragma omp parallel for private(err) shared(n,v1,v2) reduction(+:s1,s2)
  for (i=0; i<n; i++) {
    err = v1[i] - v2[i];
    s1 += err;
    s2 += err*err;
  }

  moyenne = s1/n;
  ecart_type = sqrt(s2/n - moyenne*moyenne);
}

int main(int argc, char **argv)
{
  int nthreads;
  #pragma omp parallel
  {
    #pragma omp master
    nthreads = NTHREADS;
  }

  size_t n = argc > 1 ? strtol(argv[1], nullptr, 10) : 2000;
  int imax = argc > 2 ? strtol(argv[2], nullptr, 10) : IMAX;
  set_terms(imax);

  double elapsed_init, elapsed_stat;

  std::cout << "\n\nversion OpenMP fin : \n\t" << nthreads << " thread(s)\n"
            << "\ttaille vecteur = " << n << "\n"
            << "\ttermes (formule Taylor) : " << imax
            << std::endl;

  std::vector<double> pos(n), v1, v2;
  double m, e;
  
  double t0 = omp_get_wtime();
  init(pos, v1, v2);
  elapsed_init = omp_get_wtime() - t0;

  if (n < 10000)
    save("sinus.dat", pos, v1, v2);
  
  t0 = omp_get_wtime();
  stat(v1, v2, m, e);
  elapsed_stat = omp_get_wtime() - t0;
  
  std::cout << "erreur moyenne : " << m << " ecart-type : " << e
            << std::endl << std::endl;
  
  std::cout << "time init : "
            << std::setw(12) << elapsed_init << " s" << std::endl; 
  std::cout << "time stat : "
            << std::setw(12) << elapsed_stat << " s" << std::endl;
  
  return 0;
}
