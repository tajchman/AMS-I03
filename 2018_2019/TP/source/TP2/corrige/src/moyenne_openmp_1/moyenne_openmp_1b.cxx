#include "timer.hpp"
#include "keyPress.hpp"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>

void init(std::vector<double> & v)
{
  int i, n = v.size();
  for (i=0; i<n; i++)
    v[i] = 2 + sin(i*3.14159/n);
}

void stat(const std::vector<double> & v, double & moyenne, double & ecart_type)
{
  double s1 = 0.0, s2 = 0.0;
  int i, n = v.size();
  double a = 2;
  
#pragma omp parallel for default(none) reduction(+:s1,s2)
  for (i=0; i<n; i++) {
    s1 += a*v[i];
    s2 += v[i]*v[i];
  }

  moyenne = s1/n;
  ecart_type = sqrt(s2/n - moyenne*moyenne);
}

int main(int argc, char **argv)
{
  size_t n = argc > 1 ? strtol(argv[1], nullptr, 10) : 10000000;

  std::cout << "moyenne 1 : taille vecteur = " << n << std::endl;

  Timer t_init, t_moyenne;

  t_init.start();
  std::vector<double> v(n);
  init(v);
  t_init.stop();
  
  t_moyenne.start();
  double m, e;
  stat(v, m, e);
  t_moyenne.stop();
  
  std::cout << "moyenne : " << m << " ecart-type : " << e
            << std::endl << std::endl;
  
  std::cout << "time init    : "
            << std::setw(12) << t_init.elapsed() << " s" << std::endl; 
  std::cout << "time moyenne : "
            << std::setw(12) << t_moyenne.elapsed() << " s" << std::endl;
  
  return 0;
}
