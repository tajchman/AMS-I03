#include <iostream>
#include "calcul.hxx"
#include "timer.hxx"
#include "affiche.hxx"

double calcul_seq(std::vector<double> & v, 
                double a, double (*f)(double, double),
                const std::vector<double> & u)
{
  size_t i, N = u.size();

  Timer T;
  T.start();

  for (i = 0; i<N; i++)
    v[i] = f(a, u[i]);

  T.stop();
  std::cout << "Calcul sequentiel     " << T.elapsed() << " s" << std::endl;
  affiche("v0", v);
  std::cout << std::endl;

  return T.elapsed();
}


