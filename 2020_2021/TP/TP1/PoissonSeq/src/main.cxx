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
  Timer T_global;
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

  if (freq > 0) C.getOutput().plot(0);

  int i;
  for (i=0; i<nsteps; i++) {
    C.solve(ksteps);
    if (freq > 0) C.getOutput().plot(i);
    }

  if (Prm.convection())
    std::cout << "convection ";
  else
    std::cout << "           ";
  if (Prm.diffusion())
    std::cout << "diffusion  ";
  else
    std::cout << "           ";

  T_global.stop();
  std::cout << "cpu time " << std::setprecision(5) << T_global.elapsed() << " s\n";
  return 0;
}
