#include <cmath>
#include <iostream>
#include "timer.hxx"
#include <time.h>
#include <ctime>

const long ITERATIONS = 10000L;


double urand()
{
  int r = std::rand();
  return double(r)/(RAND_MAX);
}


double Calcul_Pi(std::size_t n)
{
  std::srand(std::time(NULL)); 

  unsigned long p = 0;
  double u, v;
  std::size_t i, it;
  
  for (i=0; i<n; i++) {
    for (it = 0; it<ITERATIONS; it++) {
      u = urand();
      v = urand();
      p += (u*u+v*v < 1.0) ? 1 : 0;
    }
  }
  
  return ((4.0*p)/ITERATIONS)/n;
}

