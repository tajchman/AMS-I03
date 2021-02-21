#include "user.hxx"
#include <cmath>

double cond_ini(double x, double y, double z)
{
  return 0.0;
}

double cond_lim(double x, double y, double z)
{
  return 2.0;
}

double force(double x, double y, double z)
{
  return x*(1-x) * y*(1-y) * z*(1-z);
}

