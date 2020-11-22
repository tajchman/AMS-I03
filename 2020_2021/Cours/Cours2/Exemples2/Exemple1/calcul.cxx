#include "calcul.hxx"
#include <cmath>
#include <iostream>
#include "timer.hxx"

#include <unistd.h>

double f(double a, double x)
{
    usleep(100);
    return sin(a*x);
}

void calcul_seq(std::vector<double> & v, 
                const std::vector<double> & u)
{
  size_t i, N = u.size();
  double a = M_PI;

  Timer T;
  T.start();

  for (i = 0; i<N; i++)
    v[i] = f(a, u[i]);

  T.stop();
  std::cout << "Calcul sequentiel     " << T.elapsed() << " s" << std::endl;
}

void calcul_par0(std::vector<double> & v, 
                 const std::vector<double> & u)
{
  size_t i, N = u.size(), N0 = 0, N1=N/3, N2=2*N/3, N3=N;
  double a = M_PI;

  Timer T;
  T.start();

#pragma omp parallel sections default(shared)
  {
  #pragma omp section
    {
      for (i = N0; i<N1; i++)
        v[i] = f(a, u[i]);
    }
  #pragma omp section
    {
      for (i = N1; i<N2; i++)
        v[i] = f(a, u[i]);
    }
  #pragma omp section
    {
      for (i = N2; i<N3; i++)
        v[i] = f(a, u[i]);
    }
  }

  T.stop();
  std::cout << "Calcul parallele (v1) " << T.elapsed() << " s" << std::endl;
}

void calcul_par1(std::vector<double> & v, 
                 const std::vector<double> & u)
{
  size_t i0, i1, i2, N = u.size(), N0 = 0, N1=N/3, N2=2*N/3, N3=N;
  double a = M_PI;

  Timer T;
  T.start();

#pragma omp parallel sections
  {
  #pragma omp section
    {
      for (i0 = N0; i0<N1; i0++)
        v[i0] = f(a, u[i0]);
    }
  #pragma omp section
    {
      for (i1 = N1; i1<N2; i1++)
        v[i1] = f(a, u[i1]);
    }
  #pragma omp section
    {
      for (i2 = N2; i2<N3; i2++)
        v[i2] = f(a, u[i2]);
    }
  }

  T.stop();
  std::cout << "Calcul parallele (v2) " << T.elapsed() << " s" << std::endl;
}

void calcul_par2(std::vector<double> & v, 
                 const std::vector<double> & u)
{
  size_t i, N = u.size();
  double a = M_PI;

  Timer T;
  T.start();

  #pragma omp parallel for
  for (i = 0; i<N; i++)
    v[i] = f(a, u[i]);

  T.stop();
  std::cout << "Calcul parallele (v3) " << T.elapsed() << " s" << std::endl;
}
