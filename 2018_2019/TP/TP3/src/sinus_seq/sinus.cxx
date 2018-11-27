#include "timer.hpp"
#include "keyPress.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>

#include "sin.hxx"

void init(std::vector<double> & pos,
          std::vector<double> & v1,
          std::vector<double> & v2)
{
  double x, pi = 3.14159265;
  int i, n = pos.size();

  v1.resize(n);
  v2.resize(n);

  for (i=0; i<n; i++) {
    x = i*2*pi/n;
    pos[i] = x;
    v1[i] = sinus_machine(x);
    v2[i] = sinus_taylor(x);
  }
}

void save(const char *filename,
	  std::vector<double> & pos,
	  std::vector<double> & v1,
	  std::vector<double> & v2)
{
  std::ofstream f(filename);

  f  << "# x sin(systeme) approximation" << std::endl;
  int i, n = pos.size();
  for (i=0; i<n; i++)
    f << pos[i] << " " << v1[i] << " " << v2[i] << std::endl;
}

void stat(const std::vector<double> & v1,
          const std::vector<double> & v2,
          double & moyenne, double & ecart_type)
{
  double s1 = 0.0, s2 = 0.0, err;
  int i, n = v1.size();
  for (i=0; i<n; i++) {
    err = v1[i] - v2[i];
    s1 += err;
    s2 += err*err;
  }

  moyenne = s1/n;
  ecart_type = sqrt(s2/n - moyenne*moyenne);
}

int main(int argc, char **argv)
{
  size_t n = argc > 1 ? strtol(argv[1], nullptr, 10) : 2000;
  int imax = argc > 2 ? strtol(argv[2], nullptr, 10) : IMAX;
  set_terms(imax);

  std::cout << "\n\nversion sequentielle : \n"
            << "\ttaille vecteur = " << n << "\n"
            << "\ttermes (formule Taylor) : " << imax
            << std::endl;

  Timer t_init, t_stat;

  std::vector<double> pos(n), v1, v2;
  double m, e;
  
  t_init.start();
  init(pos, v1, v2);
  t_init.stop();

  if (n < 10000)
    save("sinus.dat", pos, v1, v2);
  
  t_stat.start();
  stat(v1, v2, m, e);
  t_stat.stop();
  
  std::cout << "erreur moyenne : " << m << " ecart-type : " << e
            << std::endl << std::endl;
  
  std::cout << "time init : "
            << std::setw(12) << t_init.elapsed() << " s" << std::endl; 
  std::cout << "time stat : "
            << std::setw(12) << t_stat.elapsed() << " s" << std::endl;
  
  return 0;
}