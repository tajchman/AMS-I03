#include "calcul.hxx"
#include "timer_papi.hxx"
#include <cmath>

void calcul(std::vector<double> & v, const std::vector<double> & u)
{
  size_t i, n = u.size();

  v[0] = u[0];

  for (i = 1; i<n-1; i++) {
    Timer T;
    T.start();

    v[i] = (exp(u[i-1])+2*sin(u[i])+u[i+1])/4;

    T.stop();
    std::cerr << i << " " << T.elapsed() << std::endl;
  }

  v[n-1] = u[n-1];
}
