#include <iostream>
#include <sstream>
#include <iomanip>

#include "parameters.hxx"
#include "values.hxx"
#include "scheme.hxx"
#include "timer.hxx"

double f(double x, double y, double z)
{
  if (x*x+y*y+z*z < 0.2)
    return 1.0;
  else
    return 0.0;
}

int main(int argc, char **argv)
{
  int iOut = 0;
  Timer T_init, T_comp, T_io, T_iter, T_total;
  T_total.start();
  T_init.start();

  Parameters P(&argc, &argv);
  if (P.help()) return 0;

  std::cout << P << std::endl;
  P.out() << P << std::endl;

  Values u1(P, f);
  Values u2(u1);
  T_init.stop();

  if (P.output() > 0) {
    T_io.start();
    u1.plot(iOut++);
    T_io.stop();
  }

  double dt = P.dt(), t, du;
  int it;

  for (it = 1, t=0; it <= P.itmax(); it++) {
      T_comp.start();
      T_iter.reset();
      T_iter.start();

      t += dt;
      du = iterate(u1, u2, dt, P);
      u2.swap(u1);

      T_iter.stop();
      T_comp.stop();

      std::ostringstream out;
      out << std::setw(10) << it
	  << " t = " << std::setw(12) << t
	  << " delta u = " << std::setw(10) << du
	  << " cpu time = " << std::setw(10) << T_iter.elapsed()
	  << "\n";
      std::cout << out.str();
      P.out() << out.str();
    
      if (P.output() > 0 && it % P.output() == 0) {
	T_io.start();
	u1.plot(iOut++);
	T_io.stop();
      }
  }
  
  T_total.stop();
  
  std::ostringstream out;
  out << "\n\n";
  out << "Initialization time  : " << std::setw(10) << T_init.elapsed() << "\n";
  out << "Output time          : " << std::setw(10) << T_io.elapsed() << "\n";
  out << "Computation time     : " << std::setw(10) << T_comp.elapsed() << "\n";
  out << "\nTotal cpu time       : " << std::setw(10) << T_total.elapsed() << "\n";
  std::cout << out.str();
  P.out() << out.str();
  
  return 0;
}
