#include "parameters.hxx"
#include "scheme.hxx"

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

    C.terminate();
  }

  return 0;
}
