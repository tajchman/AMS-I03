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

double verifie(const std::vector<double> & v1, 
               const std::vector<double> & v2)
{
   double diff = 0;
   int i, n = v1.size();
   for (i=0; i<n; i++)
      diff += std::abs(v1[i] - v2[i]);
   return diff;
}

int main(int argc, char **argv) {
   size_t  n = argc > 1 ? strtol(argv[1], NULL, 10) : 10000;
   std::vector<double> u(n), v0(n, 0), v1(n, 0), v2(n, 0), v3(n, 0), v4(n, 0);
   double a = M_PI;

   init(u);

   calcul_seq (v0, a, f, u);
   
   return 0;
 }
