#include <iostream>
#include <iomanip>
#include "timer.hxx"
#include "calcul.h"
#include <cstdlib>

int main(int argc, char **argv)
{
  Timer T;

  T.start();
  
  int n = argc > 1 ? strtol(argv[1], NULL, 10) : 200L;
  int it, n_it = argc > 2 ? strtol(argv[2], NULL, 10) : 20000L;
  int isave = argc > 3 ? strtol(argv[3], NULL, 10) : 0;

  std::cerr << "Calcul sur GPU : n = " << n << std::endl;
  
  double * u_current = init(n);
  double * u_next = zero(n);
  
  double * diffusion = zero(n);
  double * forces = zero(n);
  
  double * work = alloue_work(n);
  
  double delta;
  const double tol = 1e-6;

  double dx = 1.0/(n-1);
  double dt = 0.5*dx*dx;
  double t = 0.0;
  int ksave = 0;
  
  for (it = 0; it < n_it; it++) {

    t = it * dt;
    if (isave > 0 && ((it / isave) * isave == it)) {
      save(ksave, u_current, n);
      ksave ++;
    }
    laplacien(diffusion, u_current, dx, n);
    
    calcul_forces(forces, u_current, n);
    
    variation(u_next, u_current, diffusion, forces, dt, n);

    double *w = u_current;
    u_current = u_next;
    u_next = w;

    delta = difference(u_current, u_next, work, n);
    std::cerr << "iteration " << std::setw(6) << it+1 << "  delta = "
	      << std::fixed
	      << std::setw(13)
	      << std::setprecision(7)
	      << delta << "     \r";
    
    if (delta < tol) break;
  }

  std::cerr << "\n\n";

  T.stop();
  std::cout << "temps calcul : " << T.elapsed() << " s"
	    << std::endl;

  save(ksave, u_current, n);
  libere(&u_current);
  libere(&u_next);
  return 0;
}
