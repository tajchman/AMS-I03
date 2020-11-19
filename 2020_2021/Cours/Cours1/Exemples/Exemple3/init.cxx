#include "init.hxx"
#include <cstdlib>
#include <ctime>

void init(std::vector<double> &v, double v0)
{
  int i, n = v.size();

  for (i=0; i<n; i++)
      v[i] = v0;
}
