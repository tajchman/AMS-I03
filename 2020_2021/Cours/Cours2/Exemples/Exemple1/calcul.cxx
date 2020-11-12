#include "calcul.hxx"

void calcul(std::vector<double> & v, const std::vector<double> & u)
{
  v[0] = u[0];
  for (i = 1; i<n-1; i++)
    v[i] = (u[i-1]+2*u[i]+u[i+1])/4;
  v[n-1] = u[n-1];
}
