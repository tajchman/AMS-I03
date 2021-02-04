#include "calcul.hxx"
#include "timer.hxx"
#include <cmath>
#include <iostream>

void calcul(std::vector<double> & u,
            double a, const std::vector<double> & v,
            double b, const std::vector<double> & w,
            bool terme1, bool terme2)
{
  int i, n = u.size();

  Timer T1;
  T1.start();

  for (i=0; i<n; i++) {
    if (terme1) u[i] += a*v[i];
    if (terme2) u[i] += b*w[i];
  }

  T1.stop();

  Timer T2;
  T2.start();

  double aa, bb;
  if (terme1) aa = a; else aa = 0.0;
  if (terme2) bb = b; else bb = 0.0;

  for (i=0; i<n; i++) {
     u[i] += aa*v[i] + bb*w[i];
  }

  T2.stop();

  std::cout << "tests dans la boucle    " << T1.elapsed() << std::endl;
  std::cout << "tests hors de la boucle " << T2.elapsed() << std::endl;
}
