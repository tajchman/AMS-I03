#include "timer.hpp"
#include "keyPress.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>

double mon_sinus(double x)
{
  double y = x, x2 = x*x;
   int i, imax = 8, m;
   double coef = x;
   for (i=1; i<imax; i++) {
     m = 2*i*(2*i+1);
     coef *= -x2/m;
     y += coef;
   } 
   return y;
}

void init(std::vector<double> & exact, std::vector<double> & v)
{
  double x, pi = 3.14159265;
  int i, n = v.size();

  for (i=0; i<n; i++) {
    x = i*2*pi/n;
    v[i] = mon_sinus(x);
    exact[i] = sin(x);
  }
}

void save(const char *filename,
	  std::vector<double> & v1,
	  std::vector<double> & v2)
{
  std::ofstream f(filename);
  int i, n = v1.size();
  for (i=0; i<n; i++)
    f << v1[i] << " " << v2[i] << std::endl;
}

void stat(const std::vector<double> & v, double & moyenne, double & ecart_type)
{
  double s1 = 0.0, s2 = 0.0;
  int i, n = v.size();
  for (i=0; i<n; i++) {
    s1 += v[i];
    s2 += v[i]*v[i];
  }

  moyenne = s1/n;
  ecart_type = sqrt(s2/n - moyenne*moyenne);
}

int main(int argc, char **argv)
{
  size_t n = argc > 1 ? strtol(argv[1], nullptr, 10) : 10000000;

  std::cout << "version 1 : taille vecteur = " << n << std::endl;

  Timer t_init, t_moyenne;

  t_init.start();
  std::vector<double> v(n), w(n);
  init(v, w);
  t_init.stop();

  if (n < 10000)
    save("sinus.dat", v, w);
  
  t_moyenne.start();
  double m, e;
  stat(w, m, e);
  t_moyenne.stop();
  
  std::cout << "moyenne : " << m << " ecart-type : " << e
            << std::endl << std::endl;
  
  std::cout << "time init    : "
            << std::setw(12) << t_init.elapsed() << " s" << std::endl; 
  std::cout << "time moyenne : "
            << std::setw(12) << t_moyenne.elapsed() << " s" << std::endl;
  
  return 0;
}
