#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

#include "sin.hxx"

void init(std::vector<double> & pos,
          std::vector<double> & v1,
          std::vector<double> & v2,
          int n1, int n2)
{
  double x, pi = 3.14159265;
  int i, n = pos.size();

#pragma omp parallel for private(x)
  for (i=n1; i<n2; i++) {
    x = i*2*pi/n;
    pos[i] = x;
    v1[i] = sinus_machine(x);
    v2[i] = sinus_taylor(x);
  }
}

void stat(const std::vector<double> & v1,
          const std::vector<double> & v2,
          int n1, int n2,
          double & sum1, double & sum2)
{
  double s1 = 0.0, s2 = 0.0, err;
  int i;

  for (i=n1; i<n2; i++) {
    err = v1[i] - v2[i];
    s1 += err;
    s2 += err*err;
  }

  sum1 = s1;
  sum2 = s2;
}

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
