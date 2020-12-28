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

  int freq = Prm.freq();
  bool output = freq > 0;
  if (output) Prm.out();
  
  int itMax = Prm.itmax();

  int nsteps = freq > 0 ? itMax/freq : 1;
  int ksteps = freq > 0 ? freq : itMax;

  Values u_0(&Prm);
  Scheme C(&Prm);
  C.timer(0).start();
  C.initialize();

#pragma omp parallel 
  {
    u_0.init(f);

#pragma omp single
    {
      C.setInput(u_0);
      C.timer(0).stop();

      if (output) C.getOutput().plot(0);
    }

    int i;
    for (i=0; i<nsteps; i++) {
	    C.solve(ksteps);
	    if (output) C.getOutput().plot(i);
    }
  }

  T_global.stop();
  std::cout << "cpu time " << std::setprecision(5) 
            << T_global.elapsed() << " s\n";
  return 0;
}
