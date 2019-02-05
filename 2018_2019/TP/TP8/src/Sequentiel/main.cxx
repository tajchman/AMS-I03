#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

#include "timer.hxx"
#include "Matrix.hxx"
#include "heat.hxx"

double u0(double x, double y)
{
  double r = 0.001 + 200*((x-0.5)*(x-0.5) + 2*(y - 0.3)*(y-0.3));
  return 1.0 - exp(-1.0/r);
}

double h(double x, double y)
{
  return x > 0.5 ? 1.0 : 0.0;
}

int main(int argc, char **argv)
{

  int i,j,n = argc > 1 ? strtol(argv[1], NULL, 10) : 2000;
  int it, n_it = argc > 2 ? strtol(argv[2], NULL, 10) :  1000;

  Timer T_init, T_calcul, T_result;
  
  T_init.start();
    
  Matrix u(n,n), v(n,n), f(n,n);

  double diff, lambda, dt;
  
  double x, y;
  for (i=0; i<n; i++)
    for (j=0; j<n; j++) {
      x = i*1.0/(n-1);
      y = j*1.0/(n-1);
      u(i,j) = u0(x, y);
      f(i,j) = h(x, y);
    }

  v = u;
  
  T_init.stop();

  T_calcul.start();
  lambda = 0.125;
  dt = 0.5/(n*n);

  int it_output = n_it > 10000 ? n_it/1000 : 1;

  for (it = 0; it<n_it; it++) {

    Iteration(v, u, f, lambda, dt);

    diff = Difference(u, v);
        
    if (it % it_output == 0) {
      std::cout << "\rit " << std::setw(7) << it << " variation = "
		<< std::setw(12) << diff << "             ";
      save(it, v);
    }
    
    Matrix::swap(u,v);
  }
  T_calcul.stop();

  T_result.start();
  std::ofstream fOut("resultat.dat");
  fOut << u << std::endl;
  T_result.stop();
  
  std::cerr << "\n\tT init   : " << T_init.elapsed() << "s " 
	    << "\n\tT calcul : " << T_calcul.elapsed() << "s"
	    << "\n\tT sortie : " << T_result.elapsed() << "s"
	    << std::endl;
  return 0;
}
