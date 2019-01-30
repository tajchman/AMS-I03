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
  
  double * u_current = init(n);
  double * u_next = alloue(n);
  
  double * diffusion = alloue(n);
  double * forces = alloue(n);
  
  double * w;
  double delta;
  const double tol = 1e-4;

  double dx = 1.0/(n-1);
  double dt = 0.5*dx*dx;
  
  for (it = 0; it < n_it; it++) {
    
    laplacien(diffusion, u_current, dx, n);
    
    calcul_forces(forces, u_current, n);
    
    addition(u_next, u_current, diffusion, forces, dt, n);

    w = u_current;
    u_current = u_next;
    u_next = w;

    delta = difference(u_current, v_next, n);
    printf("delta = %12.5g    \n", delta);
    if (delta < tol) break;
  }

  printf("\n\n");

  T.stop();
  std::cout << "temps calcul : " << T.elapsed() << " s"
	    << std::endl;

  save("out.txt", u_current, n);
  libere(&u_current);
  libere(&u_next);
  return 0;
}
