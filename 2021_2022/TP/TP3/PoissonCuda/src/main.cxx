#define IN_MAIN

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

#include "parameters.hxx"
#include "values.hxx"
#include "scheme.hxx"
#include "timer_id.hxx"
#include "os.hxx"

int main(int argc, char *argv[])
{
  T_AllocId = AddTimer("alloc");
  T_CopyId = AddTimer("copy");
  T_InitId = AddTimer("init");
  T_IterationId = AddTimer("iteration");
  T_VariationId = AddTimer("variation");
  T_FreeId = AddTimer("free");
  AddTimer("total");

  Timer & T_total = GetTimer(-1);
  T_total.start();

  Parameters Prm(argc, argv);
  if (Prm.help()) return 0;

  std::cerr << Prm << std::endl;

  int itMax = Prm.itmax();
  int freq = Prm.freq();

  Scheme C(Prm);

  Values u_0(Prm);

  u_0.init();
  u_0.boundaries();

  Timer & T_init = GetTimer(T_InitId);
  std::cerr << "init  time  " << std::setw(10) << std::setprecision(3) << T_init.elapsed() << " s"
      << std::endl;

  C.setInput(u_0);

  Timer & T_iteration = GetTimer(T_IterationId);

  for (int it=0; it < itMax; it++) {
    if (freq > 0 && it % freq == 0) {
      C.getOutput().plot(it);
    }

    C.iteration();

    std::cerr << "iter. " << std::setw(5) << it+1
        << "  variation " << std::setw(15) << std::setprecision(9) << C.variation()
        << "  time  " << std::setw(10) << std::setprecision(3) << T_iteration.elapsed() << " s"
        << std::endl;
  }

  std::cerr << std::endl;

  if (freq > 0 && itMax % freq == 0) {
    C.getOutput().plot(itMax);
  }

  T_total.stop();

  PrintTimers(std::cerr);
  return 0;
}
