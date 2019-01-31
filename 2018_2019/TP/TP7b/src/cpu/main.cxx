#include <iostream>
#include <iomanip>
#include "timer.hxx"
#include "calcul.h"
#include <cstdlib>

int main(int argc, char **argv)
{
  Timer T;
  Timer T_init, T_dif, T_f, T_var, T_delta;
  
  T.start();
  
  int n = argc > 1 ? strtol(argv[1], NULL, 10) : 200L;
  int it, n_it = argc > 2 ? strtol(argv[2], NULL, 10) : 20000L;
  int isave = argc > 3 ? strtol(argv[3], NULL, 10) : 0;

  std::cerr << "Calcul sur CPU : n = " << n << std::endl;
  

  T_init.start();
  
  double * u_current = init(n);
  double * u_next = zero(n);
  
  double * diffusion = zero(n);
  double * forces = zero(n);
  
  double * work = alloue_work(n);

  T_init.stop();
  std::cerr << "T initialisation " << T_init.elapsed()  << "\n\n"<< std::endl;
  
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

    T_dif.start();
    laplacien(diffusion, u_current, dx, n);
    T_dif.stop();
    
    T_f.start();
    calcul_forces(forces, u_current, n);
    T_f.stop();
    
    T_var.start();
    variation(u_next, u_current, diffusion, forces, dt, n);
    T_var.stop();
    
    double *w = u_current;
    u_current = u_next;
    u_next = w;

    T_delta.start();
    delta = difference(u_current, u_next, work, n);
    T_delta.stop();
    
    //    if (it == n_it-1)
      std::cerr << "iteration" << std::setw(6) << it+1 << " delta ="
	      << std::fixed
	      << std::setw(9)
	      << std::setprecision(5)
	      << delta
	      << " Tdif "
	      << std::setw(6)
	      << std::setprecision(4)
	      << T_dif.elapsed()
	      << " Tf "
	      << std::setw(6)
	      << std::setprecision(4)
	      << T_f.elapsed()
	      << " Tvar "
	      << std::setw(6)
	      << std::setprecision(4)
	      << T_var.elapsed()
	      << " Tdl "
	      << std::setw(6)
	      << std::setprecision(4)
	      << T_delta.elapsed()
	      << "  \r";
    
    if (delta < tol) break;
  }

  std::cerr << "\n\n";

  T.stop();
  std::cout << "temps calcul : " << T.elapsed() << " s"
	    << std::endl;

  if (isave > 0)
    save(ksave, u_current, n);
  libere(&u_current);
  libere(&u_next);
  libere(&diffusion);
  libere(&forces);
  return 0;
}
