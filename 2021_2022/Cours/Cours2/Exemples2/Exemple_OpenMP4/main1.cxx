#include <iostream>
#include <cstdlib>

#include "Matrice.hxx"
#include "timer.hxx"

int main(int argc, char **argv) {
  
  Timer T_total, T_init, T_calcul, T_verif;
  T_total.start();
  
  {
  T_init.start();
  
  int i, n = argc > 1 ? strtol(argv[1], NULL, 10) : 4096;
  Matrice A(n,n), B(n,n), C(n,n);
  int j, m = A.m();
  double a, b;

  T_init.stop();
  T_calcul.start();

# pragma omp parallel for default(shared) private(j)
    for (i=0; i<n ; i++)
      for (j=0; j<m; j++)
        C(i,j) = a*A(i,j) + b*B(i,j);

  T_calcul.stop();
  T_verif.start();

  // verification
  for (i=0; i<n ; i++)
    for (j=0; j<m; j++)
      if (std::abs(C(i,j) - (a*A(i,j) + b*B(i,j))) > 1e-12) {
        std::cerr << "erreur sur la composante " << i  << " " << j<< std::endl;
        return -1;
      }
   T_verif.stop();
  }
  T_total.stop();

  std::cout << "temps init   CPU : " << T_init.elapsed() << " s" << std::endl;
  std::cout << "temps calcul CPU : " << T_calcul.elapsed() << " s" << std::endl;
  std::cout << "temps verif  CPU : " << T_verif.elapsed() << " s" << std::endl;
  std::cout << std::endl;
  std::cout << "temps total  CPU : " << T_total.elapsed() << " s" << std::endl;
  return 0;
}
