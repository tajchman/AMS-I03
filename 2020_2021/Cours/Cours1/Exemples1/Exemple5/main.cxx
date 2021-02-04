#include "calcul.hxx"
#include <cstdlib>

int main(int argc, char **argv) {

  int n = argc > 1 ? strtol(argv[1],  NULL, 10) : 1000000;
  bool t1 = argc > 2 ? (argv[2][0] == '1') : true;
  bool t2 = argc > 3 ? (argv[3][0] == '1') : true;
  
  double a = 3.3, b = 1.4;
  std::vector<double> u(n, 0.0), v(n, 1.1), w(n, 2.2);

  
  calcul(u, a, v, b, w, t1, t2);

  return 0;
 }
