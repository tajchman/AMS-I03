#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

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
     return 1.0;
//     return sin(x-0.5) * exp(- y*y);
}

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

  Scheme C(Prm, force);
  C.initialize();
 
  Values u_0(Prm);
  u_0.boundaries(cond_ini);

#pragma omp parallel 
  {
  u_0.init(cond_ini);

  #pragma omp single
  {
  C.setInput(u_0);
  T_init.stop();
  std::cout << "\n  temps init "  << std::setw(10) << std::setprecision(6) 
            << T_init.elapsed() << " s\n" << std::endl;
  }

  for (int it=0; it < itMax; it++) {
    #pragma omp single
    {
    if (freq > 0 && it % freq == 0) {
      T_other.start();
      C.getOutput().plot(it);
      T_other.stop();
      }
    C.initIteration();
    }

    T_calcul.start();
    C.iteration();
    T_calcul.stop();

    #pragma omp single
    {
    std::cout << "iteration " << std::setw(5) << it 
              << "  variation " << std::setw(10) << C.variation()
              << "  temps calcul " << std::setw(10) << std::setprecision(6) 
              << T_calcul.elapsed() << " s"
              << std::endl; 
    }
  }

  if (freq > 0 && itMax % freq == 0) {
    T_other.start();
    C.getOutput().plot(itMax);
    T_other.stop();
    }
  }

  C.terminate();

  T_total.stop();

  std::cout << "\n" << std::setw(26) << "temps total" 
            << std::setw(10) << T_total.elapsed() << " s\n" << std::endl;

  #ifdef _OPENMP
    int id = Prm.nthreads();
  #else
    int id = 0;
  #endif

  std::string s = Prm.resultPath();
  s += "/temps_";
  s += std::to_string(id) + ".dat";
  std::ofstream f(s.c_str());
  f << id << " " << T_total.elapsed() << std::endl;

  return 0;
}
