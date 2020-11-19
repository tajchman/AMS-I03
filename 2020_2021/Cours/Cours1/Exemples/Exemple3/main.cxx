#include "init.hxx"
#include "calcul.hxx"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include "Matrice.hxx"
#include "timer.hxx"

#define SIZE 100000000
int main(int argc, char **argv) {
 
   int n = (argc > 1) ? strtol(argv[1], NULL, 10) : 10;
   Matrice A(n,n), B(n,n), C(n,n);

   std::cout << "Addition de matrices (" << n << "," << n << ")" << std::endl;

   Timer T;
   T.start();
      
   addition1(u, k);
    
   T.stop();
   std::cerr << "Addition 1 : " << T.elapsed() << std::endl;

   return 0;
 }
