#include <iostream>
#include "calcul.hxx"
#include "timer.hxx"
#include "affiche.hxx"
#include "verifie.hxx"

double calcul_par1(std::vector<double> & v, 
                 double a, double (*f)(double, double),
                 const std::vector<double> & u,
                 const std::vector<double> & v_seq
                )
{
  size_t i0, i1, i2, N = u.size(), N0 = 0, N1=N/3, N2=2*N/3, N3=N;

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
  std::cout << "Calcul parallele (v1) " << T.elapsed() << " s" << std::endl;
  if (N < 11) affiche("v1", v);
  verifie(v_seq, v); 
  return T.elapsed();
}
