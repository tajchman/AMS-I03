#include <iostream>
#include <sstream>
#include <iomanip>

#include "parameters.hxx"
#include "values.hxx"
#include "gpu_scheme.hxx"
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

template<typename SchemeType>
void run(Parameters & Prm) {
	Timer T_global;
	T_global.start();

	int freq = Prm.freq();
	bool output = freq > 0;
	if (output) Prm.out();

	int itMax = Prm.itmax();

	int nsteps = output ? itMax/freq : 1;
	int ksteps = output ? freq : itMax;

	Values u_0(&Prm);
	SchemeType C(&Prm);

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

	T_global.stop();

	std::cout << C.deviceName << " time " << std::setprecision(5) << T_global.elapsed() << " s\n\n";
}

int main(int argc, char *argv[])
{
  Parameters Prm(argc, argv);
  if (Prm.help()) return 0;
  std::cout << Prm << std::endl;

  run<Scheme>(Prm);
  run<GPUScheme>(Prm);

  return 0;
}
