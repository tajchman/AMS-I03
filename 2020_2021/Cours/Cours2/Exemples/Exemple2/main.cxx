#include "init.hxx"
#include "calcul.hxx"
#include <iostream>
#include <vector>

int main() {
   size_t  n = 10000000;
   std::vector<double> u(n), v(n);

   init(u);

   for (int it = 0; it<100; it++)
   {
      calcul(v, u);
      u = v;
   }

   return 0;
 }
