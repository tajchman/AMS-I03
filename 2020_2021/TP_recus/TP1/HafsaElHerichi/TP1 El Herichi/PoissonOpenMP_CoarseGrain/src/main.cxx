#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

#include <omp.h>

#include "parameters.hxx"
#include "values.hxx"
#include "scheme.hxx"
#include "timer.hxx"
#include "os.hxx"

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
  if (x < 0.3)
    return 0.0;
  else
    return sin(x-0.5) * exp(- y*y);
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
  
#ifdef _OPENMP
  int nThreads = Prm.nthreads();
#else
  int nThreads = 0;
#endif
  
  T_init.start();

#ifdef _OPENMP
  omp_set_num_threads(nThreads);
#endif
  
  Scheme C(Prm, force);
  Values u_0(Prm);
 
 #pragma omp parallel 
 {
  u_0.boundaries(cond_ini);
  u_0.init(cond_ini);
  }
  
  #pragma omp single
  {
  C.setInput(u_0);
 
  T_init.stop();
  std::cout << "\n  temps init "  << std::setw(10) << std::setprecision(6) 
            << T_init.elapsed() << " s\n" << std::endl;        
 }//end pragma omp single

   #ifdef _OPENMP
   int iThread = omp_get_thread_num();
   int dn=freq/nThreads;
   int n1;
   int n2;
   for (int i=0; i<3 ; i++)
   {
     n1 = Prm.imin_local(i,iThread); 
     n2 = Prm.imax_local(i,iThread); 
   }
  #endif
  
  #pragma omp parallel for 
    for (int it=0; it < itMax; it++) {
      #ifdef _OPENMP
        for (int iX = n1 ; iX < n2 ; iX ++)
          if (freq > 0 && iX % freq == 0) {
            T_other.start();
            C.getOutput().plot(iX);
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
          
          for (int iX = n1 ; iX < n2 ; iX ++)
           if (freq > 0 && n2 % freq == 0) {
            T_other.start();
            C.getOutput().plot(n2);
            T_other.stop();
        } 
        
      #else
        if (freq > 0 && it % freq == 0) {
            T_other.start();
            C.getOutput().plot(it);
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
              
           if (freq > 0 && itMax % freq == 0) {
            T_other.start();
            C.getOutput().plot(itMax);
            T_other.stop();
        }  
      #endif
  }//end for

  T_total.stop();

  std::cout << "\n" << std::setw(26) << "temps total" 
            << std::setw(10) << T_total.elapsed() << " s\n" << std::endl;

#ifdef _OPENMP
  int id = Prm.nthreads();
#else
  int id = 0;
#endif

  std::string s = Prm.resultPath();
  mkdir_p(s.c_str());
  s += "/temps_";
  s += std::to_string(id) + ".dat";
  std::ofstream f(s.c_str());
  f << id << " " << T_total.elapsed() << " " << C.variation() << std::endl;

  return 0;
}
