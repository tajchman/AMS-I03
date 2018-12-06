#include <cmath>
#include <iostream>
#include "calcul.hxx"
#include "timer.hxx"

Calcul_CPU::Calcul_CPU(std::size_t n)
{
  Timer T1; T1.start();
  
  h_u.resize(n);
  h_v.resize(n);
  h_w.resize(n);
  
  T1.stop();
  std::cerr << "\t\ttemps init 1 : " << T1.elapsed() << std::endl;
  Timer T2; T2.start();

  std::size_t i;
  double x;
  
  for( i = 0; i < n; i++ ) {
    x = double(i);
    h_u[i] = sin(x)*sin(x);
    h_v[i] = cos(x)*cos(x);
  }

  T2.stop();
  std::cerr << "\t\ttemps init 2 : " << T2.elapsed() << std::endl;
}

void Calcul_CPU::addition()
{
  Timer T; T.start();
  
  std::size_t i, n = h_u.size();
  for (i=0; i<n; i++)
    h_w[i] = h_u[i] + h_v[i];
  
  T.stop();
  std::cerr << "\t\ttemps add.   : " << T.elapsed() << std::endl;
}

double Calcul_CPU::verification()
{
  Timer T; T.start();
  
  std::size_t i, n = h_u.size();

  double s = 0;
  for (i=0; i<n; i++)
    s += h_w[i];
  s = s/n - 1.0;
  
  T.stop();
  std::cerr << "\t\ttemps verif. : " << T.elapsed() << std::endl;

  return s;
}

