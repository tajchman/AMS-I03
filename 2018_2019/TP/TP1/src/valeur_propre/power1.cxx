#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include "Matrice.hxx"
#include "Vecteur.hxx"
#include "util.h"
#include "timer.hxx"

void init(Matrice &A, Vecteur & V)
{
  int i, j, n = V.size();
  
  std::srand(std::time(NULL));
  
  for (i=0; i<n; i++)
    V(i) = std::rand();
  V.normalise();

  for (i=0; i<n; i++)
    for (j=0; j<n; j++)
      A(i,j) = 1.0/n;
  
  for (i=0; i<n; i++)
    A(i,i) = 5 + 1.0/n;
}

void produit_matrice_vecteur(Vecteur &W, Matrice &A, Vecteur & V)
{
  int n = A.n(),i,j;
  double s;
  
  W = 0.0;
  
  for (j=0; j<n; j++) {
    s = V(j);
    for (i=0; i<n; i++)
      W(i) += A(i,j) * s;
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

  int n = argc > 1 ? strtol(argv[1], NULL, 10) : 3000;

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
    
    double lambda = 0.0, lambda0;
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
