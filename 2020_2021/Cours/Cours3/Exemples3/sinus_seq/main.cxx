#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

#include "init.hxx"
#include "stat.hxx"
#include "sin.hxx"

int main(int argc, char **argv)
{
  size_t n = argc > 1 ? strtol(argv[1], NULL, 10) : 2000;
  int imax = argc > 2 ? strtol(argv[2], NULL, 10) : IMAX;
  set_terms(imax);
     
  std::vector<double> pos(n), v1(n), v2(n);
   
  init(pos, v1, v2, 0, n);

  double m, e;
  
  stat(v1, v2, 0, n, m, e);

  m = m/n;
  e = sqrt(e/n - m*m);
  std::cout << "m = " << m << " e = " << e << std::endl;
  
  return 0;
}
