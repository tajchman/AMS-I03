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
  Parameters Prm(&argc, &argv);
  if (Prm.help()) return 0;
  if (Prm.rank() == 0)
    std::cout << Prm << std::endl;

  int freq = Prm.freq();
  bool output = freq > 0;
  int itMax = Prm.itmax();

  int nsteps = freq > 0 ? itMax/freq : 1;
  int ksteps = freq > 0 ? freq : itMax;

  {
    Scheme C(&Prm);
    Values u_0(&Prm, f);

    C.timer(0).start();

    C.setInput(u_0);

    C.timer(0).stop();

    if (output) C.getOutput().plot(0);

    int i;
    for (i=0; i<nsteps; i++) {
	C.solve(ksteps);
	if (output) C.getOutput().plot(i);
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

if (P.rank() == 0) (
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
)
  return 0;
}
