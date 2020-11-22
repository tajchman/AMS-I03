#include <iostream>
#include "calcul.hxx"
#include "timer.hxx"
#include "affiche.hxx"

void calcul_par2(std::vector<double> & v, 
                 double a, double (*f)(double, double),
                 const std::vector<double> & u)
{
  size_t i, N = u.size(), N0 = 0, N1=N/3, N2=2*N/3, N3=N;

  Timer T;
  T.start();

#pragma omp parallel sections private(i)
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
  std::cout << "Calcul parallele (v2) " << T.elapsed() << " s" << std::endl;
  if (N < 10) affiche("v2", v);
}
