#include <iostream>
#include <sstream>
#include <iomanip>

#include "parameters.hxx"
#include "values.hxx"
#include "scheme.hxx"
#include "timer.hxx"

double f(double x, double y, double z)
{
  x = x - 0.5;
  y = y - 0.4;
  z = z - 0.6;
  
  if (x*x+y*y+z*z < 0.1)
    return 1.0;
  else
    return 0.0;
}

int main(int argc, char **argv)
{
  int iOut = 0;
  int nThreads = 1;

  Parameters P(argc, argv);
  if (P.help()) return 0;

  std::cout << P << std::endl;
  if (P.output() > 0)
    P.out() << P << std::endl;

  Timer T;
  T.start();
  Values u1(P, f);
  Values u2(u1);
  T.stop();
  std::cout << "initialisation " << T.elapsed() << " s" << std::endl;
  
  Timer T2, T3;
  
  if (P.output() > 0) {
    T3.start();
    u1.plot(iOut++);
    T3.stop();
  }

  double dt = P.dt(), t = 0, du;
  int it;

#pragma omp parallel
  {
    double du_local;
    int it;
    for (it = 1; it <= P.itmax(); it++) {
      
#pragma omp single
      {
	T2.start();
	t += dt;
	du = 0.0;
      }
      
      du_local = iterate(u1, u2, dt, P);

#pragma omp atomic
      du += du_local;
      
#pragma omp single
      {
	u2.swap(u1);
	T2.stop();
	
	if (P.output() > 0 && it % P.output() == 0) {
	  T3.start();
	  u1.plot(iOut++);
	  T3.stop();
	}

	std::ostringstream out;
	out << std::setw(8) << it
	    << " t = " << std::setw(10) << std::setprecision(5)
	    << std::fixed << t
	    << " delta u = " << std::setw(10) << std::setprecision(5)
	    << std::fixed << du << " (cpu: " << std::setprecision(5)
	    << T2.elapsed() << "s, i/o: "<< T3.elapsed() << "s)";
	std::cout << out.str() << "\r";
	std::cout.flush();
	if (P.output() > 0)
	  P.out() << out.str() << "\n";
      }
    }
  }
  std::cout << "\n\n";
  if (P.output() > 0)
    std::cout << "\n\nresults in " << P.resultPath() << "\n\n";
  
  return 0;
}
