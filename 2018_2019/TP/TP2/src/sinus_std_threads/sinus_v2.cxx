#include "timer.hxx"
#include "keyPress.hxx"
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

class init_partiel {
public:
  init_partiel(std::vector<double> & pos,
               std::vector<double> & v1,
               std::vector<double> & v2,
               int ithread, int nthread)
    : m_pos(pos), m_v1(v1), m_v2(v2) {
    m_n = pos.size();
    int dn = m_n/nthread;
    m_i1 = dn * ithread;
    m_i2 = dn *(ithread+1);
    m_pi = 3.14159265;
  }

  void operator()() {
    int i;

    for (i=m_i1; i<m_i2; i++) {
      m_pos[i] = i*2*m_pi/m_n;
      m_v1[i] = sinus_machine(m_pos[i]);
      m_v2[i] = sinus_taylor(m_pos[i]);
    }
  }
  
  std::vector<double> & m_pos;
  std::vector<double> & m_v1;
  std::vector<double> & m_v2;
  int m_i1, m_i2, m_n;
  double m_pi;
};

void init(std::vector<double> & pos,
          std::vector<double> & v1,
          std::vector<double> & v2,
          int nthreads=1) {

  int i, n = pos.size();
  
  v1.resize(n);
  v2.resize(n);
  
  std::vector<std::thread> threads(nthreads);
 
  for (i=0; i<nthreads; i++)
    threads[i] = std::thread(init_partiel(pos, v1, v2, i, nthreads));
    
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
  stat(v1, v2, m, e);
  t_moyenne.stop();
  
  std::cout << "erreur moyenne : " << m << " ecart-type : " << e
            << std::endl << std::endl;
  
  std::cout << "time init    : "
            << std::setw(12) << t_init.elapsed() << " s" << std::endl; 
  std::cout << "time moyenne : "
            << std::setw(12) << t_moyenne.elapsed() << " s" << std::endl;
  
  return 0;
}
