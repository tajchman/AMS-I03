#include "calcul.hxx"
#include <cmath>
#include <fstream>

void calcul(std::vector<double> & u, int step)
{
  size_t i, n = u.size();

  for (i = step; i<n; i += step) {
    u[i] = u[i-step] + 1;
  }
}
