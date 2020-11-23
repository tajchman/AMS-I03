#include <iostream>
#include <vector>
#include <cstdlib>
#include "timer.hxx"

int main(int argc, char **argv) {
  
  int i, n = argc > 1 ? strtol(argv[1], NULL, 10) : 100000000;
  std::vector<double> u(n, 1), v(n, 2), w(n, 0);
  double a, b;

  Timer T;
  T.start();

# pragma omp parallel shared (u ,v ,w ,a ,b ,n) private (i)
  {
# pragma omp for
    for (i=0; i < n ; i++)
      w[i] = a * u[i] + b * v[i];
  }

  T.stop();
  std::cout << "temps CPU : " << T.elapsed() << std::endl;

  // verification
  for (i=0; i < n ; i++)
    if (std::abs(w[i] - a * u[i] - b * v[i]) > 1e-12) {
      std::cerr << "erreur sur la composante " << i << std::endl;
      return -1;
    }
  return 0;
}
