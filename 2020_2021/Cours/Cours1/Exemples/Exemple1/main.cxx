#include "init.hxx"
#include "calcul.hxx"
#include <iostream>
#include <vector>
#include <cstdlib>

int main(int argc, char **argv) {

   size_t n = argc > 1 ? strtol(argv[1], nullptr, 10) : 10;
   std::vector<double> u(n), v(n);

   init(u);

   calcul(v, u);

   return 0;
 }
