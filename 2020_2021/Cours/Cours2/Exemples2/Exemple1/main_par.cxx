#include "init.hxx"
#include "calcul.hxx"
#include <iostream>
#include <vector>
#include <cmath>
#include <unistd.h>

double f(double a, double x)
{
    usleep(100);
    return sin(a*x);
}

std::string verifie(const std::vector<double> & v1, 
               const std::vector<double> & v2)
{
   double diff = 0;
   int i, n = v1.size();
   for (i=0; i<n; i++)
      diff += std::abs(v1[i] - v2[i]);
   if (diff < 1e-15)
      return "oui";
   else
      return "non";
}

int main(int argc, char **argv) {
   size_t  n = argc > 1 ? strtol(argv[1], NULL, 10) : 1000;
   std::vector<double> u(n), v0(n, 0), v1(n, 0), v2(n, 0), v3(n, 0), v4(n, 0);
   double a = M_PI;

   init(u);

   calcul_seq (v0, a, f, u);

   calcul_par0(v1, a, f, u);
   std::cout << "verification " << verifie(v0, v1)  << std::endl<< std::endl;
   
   calcul_par1(v2, a, f, u);
   std::cout << "verification " << verifie(v0, v2) << std::endl << std::endl;
   
   calcul_par2(v3, a, f, u);
   std::cout << "verification " << verifie(v0, v3) << std::endl << std::endl;
   
   calcul_par3(v4, a, f, u);
   std::cout << "verification " << verifie(v0, v4) << std::endl << std::endl;
   
   return 0;
 }
