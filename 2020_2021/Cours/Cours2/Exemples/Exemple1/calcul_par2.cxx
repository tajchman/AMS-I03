#include <iostream>
#include "calcul.hxx"
#include "timer.hxx"

void calcul_par2(std::vector<double> & v, 
                 double a, double (*f)(double, double),
                 const std::vector<double> & u)
{
  size_t i, N = u.size();

  Timer T;
  T.start();

  #pragma omp parallel for
  for (i = 0; i<N; i++)
    v[i] = f(a, u[i]);

  T.stop();
  std::cout << "Calcul parallele (v3) " << T.elapsed() << " s" << std::endl;
}
