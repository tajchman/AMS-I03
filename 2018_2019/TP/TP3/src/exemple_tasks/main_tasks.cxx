#include <list>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#define NTHREADS omp_get_num_threads()
#define ITHREAD  omp_get_thread_num()
#else
#define NTHREADS 1
#define ITHREAD  0
#endif

#include "sin.hxx"

void add(std::list<double> & L, double x)
{
}

int main()
{
  std::srand(0);
  
  std::list<std::pair<double, double>> L;
  double x = 0.001;
  
  for (x = 0.01; x < 10.0; x *= (1.0 + 0.2*double(std::rand())/RAND_MAX)) {
    L.push_back(std::pair<double, double>{x, 0.0});
  }
  std::cerr << "Liste de " << L.size() << " elements" << std::endl;
  
  set_terms(20);

  double t0 = omp_get_wtime();
 
#pragma omp parallel
  {
#pragma omp master
    
    for (auto e = L.begin(); e != L.end(); e++)
#pragma omp task 
      {
        e->second = sinus_taylor(e->first);
      }
      
#pragma omp taskwait
  }

  std::cerr << "temps de calcul : " <<  omp_get_wtime() - t0 << std::endl;

  double erreur = 0.0;
  for (const auto & e:L)
    erreur += e.second - sin(e.first);

  std::cerr << "erreur = " << std::setw(12) <<  erreur << std::endl;
  return 0;
}
