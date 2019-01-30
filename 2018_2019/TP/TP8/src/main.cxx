#include <iostream>
#include "timer.hxx"
#include "calcul.h"

int main(int argc, char **argv)
{
  printf("argc = %d %s\n", argc, argv[2]);
  Timer T;

  T.start();
  
  int n = argc > 1 ? strtol(argv[1], NULL, 10) : 1000L;
  int it, n_it = argc > 2 ? strtol(argv[2], NULL, 10) : 1000L;

  printf("n_it = %d\n", n_it);
  
  double * u = init(n);
  double * v = alloue(n);
  double * w;
  double delta;
  const double tol = 1e-4;
  
  double dt = 0.5/(n*n);
  for (it = 0; it < n_it; it++) {
    iteration(v, u, dt, n);

    w = u;
    u = v;
    v = w;

    delta = difference(u, v, n);
    printf("delta = %12.5g    \n", delta);
    if (delta < tol) break;
  }

  printf("\n\n");

  T.stop();
  std::cout << "temps calcul : " << T.elapsed() << " s"
	    << std::endl;

  save("out.txt", u, n);
  libere(&u);
  libere(&v);
  return 0;
}
