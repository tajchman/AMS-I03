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

void affiche(const char *message, const std::vector<double> & v)
{
   std::cout << message;
   for (int i=0; i<v.size(); i++)
     std::cout << " " << v[i];
   std::cout << std::endl;
}

int main(int argc, char **argv) {
   size_t  n = argc > 1 ? strtol(argv[1], NULL, 10) : 10000000;
   std::vector<double> u(n), v0(n, 0), v1(n, 0), v2(n, 0), v3(n, 0);

   init(u);

   calcul_seq (v0, u);
   std::cout << "verification " << verifie(v0, v0) << std::endl;
   if (n < 10) affiche("v0", v0);

   calcul_par0(v1, u);
   std::cout << "verification " << verifie(v0, v1) << std::endl;
   if (n < 10) affiche("v1", v1);
   
   calcul_par1(v2, u);
   std::cout << "verification " << verifie(v0, v2) << std::endl;
   if (n < 10) affiche("v2", v2);
   

   calcul_par2(v3, u);
   std::cout << "verification " << verifie(v0, v3) << std::endl;
   if (n < 10) affiche("v3", v3);
   
   return 0;
 }
