#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <ctime>
#include "Matrice.hpp"
#include "Vecteur.hpp"

void init(Matrice &a, Vecteur & v)
{
  int i, j, n = v.size();
  
  std::srand(std::time(nullptr));
  
  for (i=0; i<n; i++)
    v(i) = std::rand();
  v.normalise();

  for (i=0; i<n; i++)
    for (j=0; j<n; j++)
      a(i,j) = 1.0/n;
  
  for (i=0; i<n; i++)
    a(i,i) = 5 + 1.0/n;
}

double variation(double a, double b)
{
  return std::abs(a-b)/(std::abs(a) + std::abs(a) + 1.0);
}


int main(int argc, char **argv)
{
  int i, j;
  int n = argc > 1 ? strtol(argv[1], nullptr, 10) : 1000;

  Matrice a(n,n);
  Vecteur v(n), w(n);

  init(a, v);

  double s, lambda = 0.0, lambda0;
  int k, kmax = 100;
  for(k=0; k < kmax; k++) {

    lambda0 = lambda;
    
    for (i=0; i<n; i++)
      w(i) = 0;
      
    for (j=0; j<n; j++) {
      s = v(j);
      for (i=0; i<n; i++)
	w(i) += a(i,j) * s;
    }
    lambda = w.normalise();
    v = w;
    
    std::cerr << std::setw(5) << k
	      << std::setw(15) << std::setprecision(10) << lambda
	      << '\r';

    if (variation(lambda,lambda0) < 1e-12)
      break;
  }
  std::cerr << std::endl;
  return 0;
}
