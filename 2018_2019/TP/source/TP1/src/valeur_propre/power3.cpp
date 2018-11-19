#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <ctime>
#include "Matrice.hpp"
#include "Vecteur.hpp"
#include "Util.h"
#include "timer.hpp"

void init(Matrice &A, Vecteur & V)
{
  int i, j, n = V.size();
  
  std::srand(std::time(nullptr));
  
  for (i=0; i<n; i++)
    V(i) = std::rand();
  V.normalise();

  for (i=0; i<n; i++)
    for (j=0; j<n; j++)
      A(i,j) = 1.0/n;
  
  for (i=0; i<n; i++)
    A(i,i) = 5 + 1.0/n;
}

inline
double dot(double *a, double *b, int n, int nmax)
{
  int l = 0;
  double s = 0;
  for (; l<nmax; l+=4) {
    s += a[l] * b[l];
    s += a[l+1] * b[l+1];
    s += a[l+2] * b[l+2];
    s += a[l+3] * b[l+3];
  }
  for (; l<n; l++)
     s += a[l] * b[l];
  return s;
}

void produit_matrice_vecteur(Vecteur &W, Matrice &A, Vecteur & V)
{
  int n = A.n(),i,j, nmax = (n/4) * 4;

  double * pV = &(V(0));
  double * pA = &(A(0, 0));

  for (i=0; i<n; i++) 
    W(i) = dot(pA + i*n, pV, n, nmax);
}

double variation(double a, double b)
{
  return std::abs(a-b)/(std::abs(a) + std::abs(a) + 1.0);
}


int main(int argc, char **argv)
{
  Timer t_total;
  t_total.start();

  int i, j;
  int n = argc > 1 ? strtol(argv[1], nullptr, 10) : 3000;

  {
    Timer t;
    t.start();

    Matrice A(n,n);
    Vecteur V(n), W(n);

    init(A, V);

    t.stop();
    std::cerr << "init    time : " << t.elapsed() << " s" << std::endl;

    t.reinit();
    t.start();
    
    double s, lambda = 0.0, lambda0;
    int k, kmax = 100;
    for(k=0; k < kmax; k++) {

      lambda0 = lambda;

      produit_matrice_vecteur(W, A, V);

      lambda = W.normalise();
      V = W;

      affiche(k, lambda);

      if (variation(lambda,lambda0) < 1e-12)
        break;
    }
    std::cerr << std::endl;
    t.stop();
    std::cerr << "compute time : " << t.elapsed() << " s"  << std::endl;
  }
  t_total.stop();
  std::cerr << "total time   : " << t_total.elapsed() << " s"  << std::endl;

  return 0;
}
