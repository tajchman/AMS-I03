#include "calcul.hxx"
#include "timer_papi.hxx"
#include <cmath>
#include <fstream>

void calcul(std::vector<double> & v, const std::vector<double> & u)
{
  size_t i, n = u.size();
  
#ifdef MESURE
  std::ofstream f("results.dat");
#endif

  v[0] = u[0];

  for (i = 1; i<n-1; i++) {
#ifdef MESURE
    Timer T;
    T.start();
#endif

    v[i] = (u[i-1]+2*u[i]+u[i+1])/4;

#ifdef MESURE
    T.stop();
    f << i << " " << T.elapsed() << std::endl;
#endif
  }

  v[n-1] = u[n-1];
}
