#include "init.hxx"
#include "calcul.hxx"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "timer.hxx"

int main(int argc, char **argv) {

  int n = argc > 1 ? strtol(argv[1], NULL, 10) : 10000000;

  std::vector<double> u(n), v(n), w(n), a(n), b(n), c(n), d(n);
  init(u, 1.0);
  init(v, 2.0);
  init(a, 3.0);
  init(a, 3.0);
  init(b, 3.0);

  Timer T1;
  T1.start();
      
  calcul1(u, v, w, a, b, c, d);
    
  T1.stop();

  Timer T2;
  T2.start();
      
  calcul2(u, v, w, a, b, c, d);
    
  T2.stop();

  std::cout << "                           temps cpu" << std::endl;
  std::cout << "'mauvaise' loc. temporelle " << T1.elapsed() << " s" << std::endl;
  std::cout << "   'bonne' loc. temporelle " << T2.elapsed() << " s" << std::endl;

  return 0;
 }
