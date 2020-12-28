#include "timer.hxx"
#include "keyPress.hxx"
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sin.hxx"

void init(std::vector<double> & pos,
          std::vector<double> & v1,
          std::vector<double> & v2)
{
  double pi = 3.14159265;
  int i, n = pos.size();

  v1.resize(n);
  v2.resize(n);

  #pragma omp parallel for
  for (i=0; i<n; i++) {
    pos[i] = i*2*pi/n;
    v1[i] = sinus_machine(pos[i]);
    v2[i] = sinus_taylor(pos[i]);
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
  
  std::ofstream t("sinus.gnp");
  t << "set output 'sinus.pdf'\n"
    << "set term pdf\n"
    << "plot 'sinus.dat' using 1:2 notitle w l lw 3, 'sinus.dat' using ($1):($3+0.03) notitle w l lw 3";
  
  t << std::endl;
}

void save(const char *filename,
	  std::vector<double> & pos,
	  std::vector<double> & v1,
	  std::vector<double> & v2,
          int iproc, int nprocs)
{
  std::ofstream f(filename);

  f  << "# x sin(systeme) approximation" << std::endl;
  int i, n = pos.size();
  for (i=0; i<n; i++)
    f << pos[i] << " " << v1[i] << " " << v2[i] << std::endl;

  if (iproc == 0) {
    std::ofstream t("sinus_mpi.gnp");
    t << "set output 'sinus_mpi.pdf'\n"
      << "set term pdf\n"
      << "plot ";
    for (int i=0; i<nprocs; i++) {
      if (i > 0) t << ", ";
      t << "'sinus_" << i << ".dat' using 1:2 notitle w l lw 3";
    }
    t << std::endl;
  }
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
  int nthreads;
  #pragma omp parallel
  {
    #pragma omp single
    nthreads = omp_get_num_threads();
  }

  Timer T_total;
  T_total.start();

  size_t n = argc > 1 ? strtol(argv[1], nullptr, 10) : 20000000;
  int imax = argc > 2 ? strtol(argv[2], nullptr, 10) : IMAX;
  set_terms(imax);

  std::cout << "\n\nversion OpenMP 1 : \n\t" << nthreads << " thread(s)\n"
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

  T_total.stop();
  std::cout << "time      : "
            << std::setw(12) << T_total.elapsed() << " s" << std::endl;  
  return 0;
}
