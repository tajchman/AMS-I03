#include <iostream>
#include <sstream>
#include <iomanip>

#include "CpuParameters.hxx"
#include "GpuParameters.hxx"
#include "CpuValues.hxx"
#include "GpuValues.hxx"
#include "CpuScheme.hxx"
#include "GpuScheme.hxx"
#include "timer.hxx"

template<typename SchemeType, typename ValueType, typename ParamType>
void run(int argc, char *argv[]) {

	ParamType P(argc, argv);

	if (P.help()) exit(0);
	std::cout << P << std::endl;

	Timer T_global;
	T_global.start();

	int freq = P.freq();
	bool output = freq > 0;
	if (output) P.out();

	int itMax = P.itmax();

	int nsteps = output ? itMax/freq : 1;
	int ksteps = output ? freq : itMax;

	ValueType u_0(&P);
	SchemeType C(&P);

	C.timer(0).start();
	C.initialize();
	u_0.init_f();

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
	run<CpuScheme, CpuValues, CpuParameters>(argc, argv);
	run<GpuScheme, GpuValues, GpuParameters>(argc, argv);

	return 0;
}
