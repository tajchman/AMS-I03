#include <cmath>
#include <iostream>
#include "timer.hxx"
#include <time.h>

#define q (m / a)
#define r (m % a)

static long int seed = 1;

void set_seed(long int s)
{
  seed = s;
}

double my_rand()
{
  long a = 16807;
  long m = 2147483647;
  long hi = seed / q;
  long lo = seed % q;
  long test = a * lo - r * hi;
  if(test > 0)
    seed = test;
  else	seed = test + m;
  return (double) seed/m;
}


double Calcul_Pi(std::size_t n)
{
  unsigned long seed = (unsigned long)time(NULL);
  set_seed(seed);

  unsigned long p = 0;
  double u, v;
  std::size_t i;
  
  for (i=0; i<n; i++) {
    u = my_rand();
    v = my_rand();
    p += (u*u+v*v < 1.0) ? 1 : 0;
  }
  
  return (4.0*p)/n;
}

