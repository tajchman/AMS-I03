#include <memory>
#include <iostream>
#include <chrono>
#include "util.h"
#include "vecteur.hpp"
#include "matrice.hpp"

int main(int argc, char **argv)
{
  std::cout << time_precision() << " s" << std::endl;

  char * p;
  size_t i, j, n = argc > 1 ? strtol(argv[1], &p, 10) : 1000;

  std::cout << "Ex 2 n = " << n << std::endl << std::endl;
  
  void * t_init;

  std::cout << std::endl <<"Reservation memoire" << std::endl;

  t_init = start();
  
  Matrice M(n, n);
  Vecteur X(n), Y(n);
  
  std::cout << "cpu time = " << elapsed(t_init) << " s" << std::endl;
  stop(t_init);
  
  std::cout << std::endl << "Initialisation matrice" << std::endl;
  t_init = start();

  for(i=0; i<n; i++)
    for(j=0; j<n; j++)
      M[i][j] = i + j + 1.0;
  
  std::cout << "cpu time = " << elapsed(t_init) << " s" << std::endl;
  stop(t_init);

  
  std::cout << std::endl << "Initialisation vecteurs" << std::endl;
  t_init = start();
  
  for(i=0; i<n; i++)
    X[i] = 2.0*i;
  
  std::cout << "cpu time = " << elapsed(t_init) << " s" << std::endl;
  stop(t_init);

  
  std::cout << std::endl << "Produit matrice-vecteur" << std::endl;
  t_init = start();
  
  for(i=0; i<n; i++)
    for(j=0; j<n; j++)
      Y[i] += M[i][j] * X[j];
   
  std::cout << "cpu time = " << elapsed(t_init) << " s" << std::endl;
  stop(t_init);
   
  return 0;
}
