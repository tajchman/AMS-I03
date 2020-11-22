#include "init.hxx"
#include "calcul.hxx"
#include <iostream>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <omp.h>

double f(double a, double x)
{
    usleep(100);
    return sin(a*x);
}

int main(int argc, char **argv) {
   size_t  n = argc > 1 ? strtol(argv[1], NULL, 10) : 10000;
   std::vector<double> u(n), v_seq(n, 0), 
                       v0(n, 0), v1(n, 0), v2(n, 0), v3(n, 0);
   double a = M_PI;

   int nThreads;
   #pragma omp parallel
   {
      #pragma omp master
      nThreads = omp_get_num_threads();

   }
   std::cout << "\n" << nThreads << " threads\n" << std::endl;

   init(u);

   double T_seq  = calcul_seq (v_seq, a, f, u);

   double T_par0 = calcul_par0(v0, a, f, u, v_seq);
   std::cout << "Speedup : " << T_seq/T_par0 
             << ", efficacite : " << 100.*T_seq/(T_par0*nThreads) << " %" 
             << std::endl<< std::endl;

   double T_par1 = calcul_par1(v1, a, f, u, v_seq);
   std::cout << "Speedup : " << T_seq/T_par1 
             << ", efficacite : " << 100.*T_seq/(T_par1*nThreads) << " %" 
             << std::endl<< std::endl;
  
    double T_par2 = calcul_par2(v2, a, f, u, v_seq);
   std::cout << "Speedup : " << T_seq/T_par2 
             << ", efficacite : " << 100.*T_seq/(T_par2*nThreads) << " %" 
             << std::endl<< std::endl;
   
   double T_par3 = calcul_par3(v3, a, f, u, v_seq);
   std::cout << "Speedup : " << T_seq/T_par3 
             << ", efficacite : " << 100.*T_seq/(T_par3*nThreads) << " %" 
             << std::endl<< std::endl;
  
   return 0;
 }
