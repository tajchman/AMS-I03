#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

#include "parameters.hxx"
#include "values.hxx"
#include "scheme.hxx"
#include "timer.hxx"
#include "os.hxx"

int main(int argc, char *argv[])
{
  AddTimer("total");
  AddTimer("init");
  AddTimer("calcul");
  AddTimer("other");

  Timer & T_total = GetTimer(0);
  Timer & T_init = GetTimer(1);
  Timer & T_calcul = GetTimer(2);
  Timer & T_other = GetTimer(3);

  T_total.start();

  Parameters Prm(argc, argv);
  if (Prm.help()) return 0;

  std::cout << Prm << std::endl;

  int itMax = Prm.itmax();
  int freq = Prm.freq();

  T_init.start();

  Scheme C(Prm);
 
  Values u_0(Prm);
  u_0.boundaries();
  u_0.init();

  u_0.print(std::cout);
  
  C.setInput(u_0);
  T_init.stop();
  std::cout << "\n  temps init "  << std::setw(10) << std::setprecision(6) 
            << T_init.elapsed() << " s\n" << std::endl;

  for (int it=0; it < itMax; it++) {
    if (freq > 0 && it % freq == 0) {
      T_other.start();
      //C.getOutput().plot(it);
      T_other.stop();
    }

    T_calcul.start();
    C.iteration();
    T_calcul.stop();

    std::cout << "iteration " << std::setw(5) << it 
              << "  variation " << std::setw(10) << C.variation()
              << "  temps calcul " << std::setw(10) << std::setprecision(6) 
              << T_calcul.elapsed() << " s"
              << std::endl; 
  }

  if (freq > 0 && itMax % freq == 0) {
    T_other.start();
    //C.getOutput().plot(itMax);
    T_other.stop();
  }

  T_total.stop();

  std::cout << "\n" << std::setw(26) << "temps total" 
            << std::setw(10) << T_total.elapsed() << " s\n" << std::endl;

  int id = 0;

  std::string s = Prm.resultPath();
  mkdir_p(s.c_str());
  s += "/temps_gpu.dat";
  std::ofstream f(s.c_str());
  f << 0 << " " << T_total.elapsed() << std::endl;

  return 0;
}
