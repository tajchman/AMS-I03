#include "init.hxx"
#include "calcul.hxx"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "timer.hxx"

int main(int argc, char **argv) {

  int n = argc > 1 ? strtol(argv[1], NULL, 10) : 10000000;

  std::vector<double> y(n), x(n);
  double a, b;
  init(x, 1.0);
  init(y, 0.0);
  a = 3.0;
  b = 4.0;

  Timer T1;
  T1.start();
      
  calcul1(y, a, x, b);
    
  T1.stop();

  Timer T2;
  T2.start();
      
  calcul2(y, a, x, b);
    
  T2.stop();

  std::cout << "          temps cpu" << std::endl;
  std::cout << "calcul1 " << T1.elapsed() << " s" << std::endl;
  std::cout << "calcul2 " << T2.elapsed() << " s" << std::endl;

  return 0;
 }
