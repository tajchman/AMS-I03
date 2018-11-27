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
          std::vector<double> & v2,
          int n1, int n2)
{
  double pi = 3.14159265;
  int i, n = pos.size();

#pragma omp single
  {
    v1.resize(n);
    v2.resize(n);
  }
  
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
          double & somme1, double & somme2)
{
  int ithread = omp_get_thread_num();

  double s1 = 0.0, s2 = 0.0, err;
  int i;
  for (i=n1; i<n2; i++) {
    err = v1[i] - v2[i];
    s1 += err;
    s2 += err*err;
  }

#pragma omp atomic
  somme1 += s1;

#pragma omp atomic
  somme2 += s2;
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
  int imax = argc > 2 ? strtol(argv[2], nullptr, 10) : 10;
  set_terms(imax);

  std::cout << "\n\nversion OpenMP grossier 1 : \n\t" << nthreads << " thread(s)\n"
            << "\ttaille vecteur = " << n << "\n"
            << "\ttermes (formule Taylor) : " << imax
            << std::endl;

  std::vector<double> pos(n), v1, v2;
  double m, e;

  m = 0;
  e = 0;
  
#pragma omp parallel shared(pos, v1, v2, n)
  {
    int n1, n2, dn;
    int ithread = ITHREAD, nthreads = NTHREADS;

    dn = n/nthreads;
    
    n1 = ithread * dn;
    n2 = (ithread+1) * dn;
    if (ithread == nthreads-1) n2 = n;
    
    init(pos, v1, v2, n1, n2);

    #pragma omp single 
    if (n < 10000)
      save("sinus.dat", pos, v1, v2);
  
    stat(v1, v2, n1, n2, m, e);
    
  }

  m = m/n;
  e = sqrt(e/n - m*m);

  std::cout << "erreur moyenne : " << m << " ecart-type : " << e
            << std::endl << std::endl;
    
  return 0;
}
