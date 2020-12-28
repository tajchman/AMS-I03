#include <iostream>
#include <cstdio>
#include <sstream>
#include <iomanip>

#include "parameters.hxx"
#include "values.hxx"
#include "scheme.hxx"
#include "timer.hxx"
#include "memory_used.h"

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
  if (Prm.rank() == 0)
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
  u_0.init(f);

  C.setInput(u_0);
  C.timer(0).stop();

  if (output) C.getOutput().plot(0);

   int i;
    for (i=0; i<nsteps; i++) {
      C.solve(ksteps);
      if (output) C.getOutput().plot(i);
    }

    memory_used(u_0.size_kb() + C.size_kb());

  }
<<<<<<< HEAD:2019_2020/TP/TP4/TP4_corrige/PoissonMPI/src/main.cxx

=======
>>>>>>> 62781e041545eccbf74ed4f8f55460f39ca0625a:2019_2020/TP windows/TP4/src/PoissonMPI/main.cxx
  T_global.stop();
  if (Prm.rank() == 0)
    std::cout << "cpu time " << std::setprecision(5)
	      << T_global.elapsed() << " s\n";
  return 0;
}
