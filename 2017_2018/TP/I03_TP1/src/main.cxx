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

  Parameters P(&argc, &argv);
  if (P.help()) return 0;

  std::cout << P << std::endl;
  P.out() << P << std::endl;

  Values u1(P, f);
  Values u2(u1);

  if (P.output() > 0) {
    u1.plot(iOut++);
  }

  double dt = P.dt(), t, du;
  int it;

  for (it = 1, t=0; it <= P.itmax(); it++) {

      t += dt;
      du = iterate(u1, u2, dt, P);
      u2.swap(u1);

      std::ostringstream out;
    out << std::setw(8) << it
        << " t = " << std::setw(10) << std::setprecision(5)
          << std::fixed << t
	  << " delta u = " << std::setw(10) << std::setprecision(5)
          << std::fixed << du;
      std::cerr << out.str() << "\r";
      P.out() << out.str() << "\n";
    
      if (P.output() > 0 && it % P.output() == 0) {
	u1.plot(iOut++);
     }
  }
   
  std::cerr << "\n\nresults in " << P.resultPath() << "\nEnd of run\n\n";
  return 0;
}
