#include "timer.hpp"
#include "keyPress.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>

#include <thread>

int imax;

double sinus_taylor(double x)
{
  double y = x, x2 = x*x;
   int i, m;
   double coef = x;
   for (i=1; i<imax; i++) {
     m = 2*i*(2*i+1);
     coef *= -x2/m;
     y += coef;
     if (std::abs(coef) < 1e-12) break;
   }
   return y;
}

double sinus_machine(double x)
{
  double y = sin(x);
  return y;
}

void init_partiel(std::vector<double> & pos,
          std::vector<double> & v1,
          std::vector<double> & v2,
          int i1, int i2)
{
  double pi = 3.14159265;
  int i, n = pos.size();

  double x;
  for (i=i1; i<i2; i++) {
    x = i*2*pi/n;
    pos[i] = x ;
    v1[i] = sinus_machine(x);
    v2[i] = sinus_taylor(x);
  }
}

void init(std::vector<double> & pos,
          std::vector<double> & v1,
          std::vector<double> & v2,
          int nthreads=1) {

  int i, n = pos.size();
  
  v1.resize(n);
  v2.resize(n);
  
  std::vector<std::thread> threads(nthreads-1);
 
  int dn = n/nthreads;
  for (i=0; i<nthreads-1; i++)
    threads[i] = std::thread(&init_partiel,
                             std::ref(pos),
                             std::ref(v1),
                             std::ref(v2),
                             dn*i, dn*(i+1));
  
  init_partiel( pos, v1, v2, dn*(nthreads-1), n);
  
  for (auto& t : threads)
    t.join();
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

void stat_partiel(const std::vector<double> & v1,
                  const std::vector<double> & v2,
                  double & s1, double & s2,
                  int i1, int i2)
{
  double s1_local = 0.0, s2_local = 0.0, err;
  int i, n = v1.size();
  for (i=i1; i<i2; i++) {
    err = v1[i] - v2[i];
    s1_local += err;
    s2_local += err*err;
  }
  s1 = s1_local;
  s2 = s2_local;
}

void stat(const std::vector<double> & v1,
          const std::vector<double> & v2,
          double & moyenne, double & ecart_type,
          int nthreads=1)
{
  double s1 = 0.0, s2 = 0.0, err;
  double s1_partiel[nthreads], s2_partiel[nthreads];
  
  int i, n = v1.size();
  
  std::vector<std::thread> threads(nthreads-1);
 
  int dn = n/nthreads;
  for (i=0; i<nthreads-1; i++)
    threads[i] = std::thread(&stat_partiel,
                             std::ref(v1), std::ref(v2),
                             std::ref(s1_partiel[i]), std::ref(s2_partiel[i]),
                             dn*i, dn*(i+1));

  stat_partiel(v1, v2,
               s1, s2,
               dn*(nthreads-1), n);
  
  for (i=0; i<nthreads-1; i++) {
    threads[i].join();
    s1 += s1_partiel[i];
    s2 += s2_partiel[i];
  }

  moyenne = s1/n;
  ecart_type = sqrt(s2/n - moyenne*moyenne);
}

int main(int argc, char **argv)
{
  int nthreads = argc > 1
    ? strtol(argv[1], nullptr, 10)
    : std::thread::hardware_concurrency();
  
  size_t n = argc > 2 ? strtol(argv[2], nullptr, 10) : 2000;
  
  imax = argc > 3 ? strtol(argv[3], nullptr, 10) : 6;

  std::cout << "\n\nversion 3 : \n\t" << nthreads << " thread(s)\n"
            << "\ttaille vecteur = " << n << "\n"
            << "\ttermes (formule Taylor) : " << imax
            << std::endl;

  Timer t_init, t_moyenne;

  t_init.start();
  std::vector<double> pos(n), v1, v2;
  init(pos, v1, v2, nthreads);
  t_init.stop();

  if (n < 10000)
    save("sinus.dat", pos, v1, v2);
  
  t_moyenne.start();
  double m, e;
  stat(v1, v2, m, e, nthreads);
  t_moyenne.stop();
  
  std::cout << "erreur moyenne : " << m << " ecart-type : " << e
            << std::endl << std::endl;
  
  std::cout << "time init    : "
            << std::setw(12) << t_init.elapsed() << " s" << std::endl; 
  std::cout << "time moyenne : "
            << std::setw(12) << t_moyenne.elapsed() << " s" << std::endl;
  
  return 0;
}
