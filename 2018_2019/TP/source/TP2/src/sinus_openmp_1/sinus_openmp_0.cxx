#include "timer.hpp"
#include "keyPress.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>

void pause()
{
  volatile double z;
  int k, l, n = 1;
  
  for (k=1; k<100*n; k++)
    for (l=1; l<100*n; l++) z = k+l;
}


double exact(const double &x)
{
  double y = sin(x);
  return y;
}

double approche(const double & x)
{
  double y = x;
   int i, imax = 20, m;
   double coef = x;
   for (i=1; i<imax; i++) {
     m = 2*i*(2*i+1);
     coef *= -x*x/m;
     y += coef;
     if (std::abs(coef) < 1e-12) break;
   }
   return y;
}

void init(std::vector<double> & v1, std::vector<double> & v2)
{
  double x, pi = 3.14159;
  int i, n = v1.size();
  
#pragma omp parallel for
  for (i=0; i<n; i++) {
    x = i*2*pi/(n-1);    
    v1[i] = exact(x);
  }
  
#pragma omp parallel for
  for (i=0; i<n; i++) {
    x = i*2*pi/(n-1);
    v2[i] = approche(x);
  }
}

void save(const char *filename,
	  const std::vector<double> & v1,
	  const std::vector<double> & v2)
{
  std::ofstream f(filename);
  int i, n = v1.size();
  for (i=0; i<n; i++)
    f << v1[i] << " " << v2[i] + 0.05 << std::endl;
}

double erreur(const std::vector<double> & v1, const std::vector<double> & v2)
{
  double s = 0.0;
  int i, n = v1.size();

#pragma omp parallel for reduction(+:s)
  for (i=0; i<n; i++)
    s += std::abs(v1[i]-v2[i]);

  return s/n;
}

int main(int argc, char **argv)
{
  size_t n = argc > 1 ? strtol(argv[1], nullptr, 10) : 10000000;

  std::cout << "moyenne 1 : taille vecteur = " << n << std::endl;

  Timer t_init, t_moyenne;

  t_init.start();
  std::vector<double> v(n), w(n);
  init(v, w);
  t_init.stop();

  if (n < 10000)
    save("sinus.dat", v, w);
  
  t_moyenne.start();
  double e;
  e = erreur(v, w);
  t_moyenne.stop();
  
  std::cout << "erreur : " << e
            << std::endl << std::endl;
  
  std::cout << "time init    : "
            << std::setw(12) << t_init.elapsed() << " s" << std::endl; 
  std::cout << "time moyenne : "
            << std::setw(12) << t_moyenne.elapsed() << " s" << std::endl;
  
  return 0;
}
