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

  Parameters P(argc, argv);
  if (P.help()) return 0;

  std::cout << P << std::endl;
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

  double dt = P.dt(), t, du;
  int it;

  for (it = 1, t=0; it <= P.itmax(); it++) {

    T2.start();
    t += dt;
    du = iterate(u1, u2, dt, P);
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
    P.out() << out.str() << "\n";
    
  }
  std::cout << "\n\nresults in " << P.resultPath() << "\n\n";
   
  if (P.convection())
    std::cout << "convection ";
  else
    std::cout << "           ";
  if (P.diffusion())
    std::cout << "diffusion  ";
  else
    std::cout << "           ";

  std::cout << "cpu time " << std::setprecision(5) << T2.elapsed() << " s "
             << std::setprecision(5) << T2.elapsed()/(it-1) << " s/iteration\n";
  return 0;
}
