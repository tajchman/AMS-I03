#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <ctime>
#include "Matrice.hpp"
#include "Vecteur.hpp"
#include "util.h"
#include "timer.hpp"

void init(Matrice &A, Vecteur & V, int p)
{
  int i, j, k, l, kmax, lmax, n = A.n();
  
  std::srand(std::time(nullptr));
  
  for (i=0; i<n; i++)
    V(i) = std::rand();
  V.normalise();

  double *pA = &A(0,0);
  double s = 1.0/n;
  for ( i = 0; i < n*n; i++) *pA++ = s;
  
  for (i=0; i<n; i++)
    A(i,i) += 5.0;
}

void produit_matrice_vecteur(Vecteur &W, Matrice &A, Vecteur & V, int p)
{
  int n = A.n(),m = A.m();
  double s;

  int pmax = (p/4) * 4;
  
  int i, j, k, l, kmax, lmax;
  for ( i = 0; i < n; i++) W(i) = 0.0;
  
  for ( i = 0; i < n; i += p) {
    kmax = i + p; if (kmax > n) kmax = n;
    for (j = 0; j < m; j += p) {
      lmax = j + p; if (lmax > m) lmax = m;

      for (k = i; k < kmax; ++k) {
	s = 0.0;
        for (l = j; l < pmax; l+=4)
          s +=  A(k,l) * V(l)
	    + A(k,l+1) * V(l+1)
	    + A(k,l+2) * V(l+2)
	    + A(k,l+3) * V(l+3);
        for (; l < lmax; ++l)
	  s +=  A(k,l) * V(l);
	W(k) += s;
      }
    }
  }
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
  int p = argc > 3 ? strtol(argv[3], nullptr, 10) : 100;  

  {
    Timer t;
    t.start();

    Matrice a(n, n);
    Vecteur v(n), w(n);

    init(a, v, p);

    t.stop();
    std::cerr << "init    time : " << t.elapsed() << " s" << std::endl;

    t.reinit();
    t.start();
    
    double s, lambda = 0.0, lambda0;
    int k, kmax = 100;
    for(k=0; k < kmax; k++) {

      lambda0 = lambda;

      produit_matrice_vecteur(w, a, v, p);

      lambda = w.normalise();
      v = w;

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
