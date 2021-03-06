#include <iostream>
#include "calcul.hxx"
#include "timer.hxx"
#include "affiche.hxx"
#include "verifie.hxx"

double calcul_par3(std::vector<double> & v, 
                 double a, double (*f)(double, double),
                 const std::vector<double> & u,
                 const std::vector<double> & v_seq
                )
{
  size_t i, N = u.size();

  Timer T;
  T.start();

  #pragma omp parallel for
  for (i = 0; i<N; i++)
    v[i] = f(a, u[i]);

  T.stop();
  std::cout << "Calcul parallele (v3) " << T.elapsed() << " s" << std::endl;
  if (N < 11) affiche("v3", v);
  verifie(v_seq, v);
  return T.elapsed();
}
