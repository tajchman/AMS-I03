#include <iostream>
#include <sstream>
#include <iomanip>

#include "parameters.hxx"
#include "values.hxx"
#include "scheme.hxx"
#include "timer.hxx"

double f(double x, double y, double z)
{
  x -= 0.5;
  y -= 0.5;
  z -= 0.5;
  if (x*x+y*y+z*z < 0.1)
    return 1.0;
  else
    return 0.0;
}

int main(int argc, char *argv[])
{
  Timer T_global, T_residu;
  T_global.start();

  Parameters Prm(argc, argv);
  if (Prm.help()) return 0;
  std::cout << Prm << std::endl;

  int itMax = Prm.itmax();
  int freq = Prm.freq();
  int nsteps, ksteps;
  
  if (freq > 0) {
    nsteps = itMax/freq;
    ksteps = freq;
    Prm.out();
  }
  else {
    nsteps = 1;
    ksteps = itMax;
  }

  Values u_0(Prm);
  Scheme C(Prm);
 
  C.timer(0).start();
  C.initialize();
  u_0.init(f);

  C.setInput(u_0);
  C.timer(0).stop();

  for (int i=0; i<nsteps; i++) {
    if (freq > 0) {
      C.timer(2).start();
      C.getOutput().plot(i);
      C.timer(2).stop();
    }

    C.solve(ksteps);
  }

  if (freq > 0) {
    C.timer(2).start();
    C.getOutput().plot(nsteps);
    C.timer(2).stop();
  }
 
  T_global.stop();

  std::cout << "\n" << std::setw(26) << "total" 
            << std::setw(10) << T_global.elapsed() << " s";
  int n = C.ntimers();
  std::cout << " (times :";
  for (int i=0; i<n; i++)
    std::cout << " " << std::setw(5) << C.timer(i).name()
	            << " " << std::setw(9) << std::fixed << C.timer(i).elapsed();
  std::cout	  << ")   \n";
  return 0;
}
