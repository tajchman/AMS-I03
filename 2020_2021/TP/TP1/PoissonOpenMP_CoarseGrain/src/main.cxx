#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <omp.h>

#include "parameters.hxx"
#include "values.hxx"
#include "scheme.hxx"
#include "timer.hxx"

double cond_ini(double x, double y, double z)
{
  x -= 0.5;
  y -= 0.5;
  z -= 0.5;
  if (x*x+y*y+z*z < 0.1)
    return 1.0;
  else
    return 0.0;
}

double force(double x, double y, double z)
{
  if (x < 0.5)
     return 0.0;
  else
     return sin(x-0.5) * exp(- y*y);
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
  C.initialize(force);
  u_0.init(cond_ini);

  C.setInput(u_0);
  C.timer(0).stop();

#pragma omp parallel 
{
  for (int i=0; i<nsteps; i++) {
   #pragma omp single
    if (freq > 0) {
      C.timer(2).start();
      C.getOutput().plot(i);
      C.timer(2).stop();
    }

    C.solve(ksteps);
  }
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

  std::string s = "temps_";
  s += std::to_string(Prm.nthreads()) + ".dat";
  std::ofstream f(s.c_str());
  f << Prm.nthreads() << " " 
    << C.timer(1).elapsed() + C.timer(2).elapsed() << std::endl;

  return 0;
}
