#include <iostream>
#include <vector>
#include <cstdlib>
#include "timer.hxx"

int main(int argc, char **argv) {

  Timer T_total, T_init, T_calcul, T_verif;
  T_total.start();
  
  {
  T_init.start();
  
  int i, n = argc > 1 ? strtol(argv[1], NULL, 10) : 100000000;
  std::vector<double> u(n, 1), v(n, 2), w(n, 0);
  double a, b;

  T_init.stop();
  T_calcul.start();

# pragma omp parallel shared (u ,v ,w ,a ,b ,n) private (i)
  {
# pragma omp for
    for (i=0; i < n ; i++)
      w[i] = a * u[i] + b * v[i];
  }

  T_calcul.stop();
  T_verif.start();

  // verification
  for (i=0; i < n ; i++)
    if (std::abs(w[i] - a * u[i] - b * v[i]) > 1e-12) {
      std::cerr << "erreur sur la composante " << i << std::endl;
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
