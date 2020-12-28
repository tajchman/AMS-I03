#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>

#include "Timer.hxx"
#include "Params.hxx"
#include "Matrix.hxx"
#include "Heat.hxx"

double g(double x, double y)
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

  int i,j,it;
  int iSave = 0;
  
  sParams P(argc, argv);
  int n = P.n;
  double diff, dt;
  
  Timer T_init, T_calcul, T_result;
  
  T_init.start();
    
  Matrix u0(n, n, g, "u0"), f(n, n, h, "f");
  
  Solver S(P);
 
  T_init.stop();

  T_calcul.start();

  S.setInput(u0);
  S.setForce(f);

  dt = 1e20;
  S.setTimeStep(dt);
 
  for (it = 0; it<P.n_it; it++) {

    S.Iteration();
    
    diff = S.Difference();
        
    std::cerr << "\rit " << std::setw(7) << it
	      << " t = " << S.getTime()
	      << " variation = " << std::setw(14)
	      << std::setprecision(4) << diff
	      << "             ";
    if (P.it_output > 0 && it % P.it_output == 0)
      S.getOutput().save(iSave++);
    
    S.Shift();
  }
  T_calcul.stop();

  if (P.it_output > 0)
    S.getOutput().save(iSave);

  std::cerr << "\n\tT init   : " << T_init.elapsed() << "s " 
	    << "\n\tT calcul : " << T_calcul.elapsed() << "s"
	    << std::endl;
  return 0;
}
