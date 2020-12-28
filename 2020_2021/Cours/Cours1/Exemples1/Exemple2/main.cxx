#include "init.hxx"
#include "calcul.hxx"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "timer.hxx"

int main(int argc, char **argv) {

  int n = argc > 1 ? strtol(argv[1], NULL, 10) : 1024;

  Matrice A(n,n), B(n,n), C(n,n);

  init(A, 1);
  init(B, 2);
  init(C, 0);

  Timer T1;
  T1.start();
      
  calcul1(C, A, B);
    
  T1.stop();

  Timer T2;
  T2.start();
      
  calcul2(C, A, B);
    
  T2.stop();

  std::cout << "                           temps cpu" << std::endl;
  std::cout << "Algo suivant les lignes   " << T1.elapsed() << " s" << std::endl;
  std::cout << "Algo suivant les colonnes " << T2.elapsed() << " s" << std::endl;

  return 0;
 }
