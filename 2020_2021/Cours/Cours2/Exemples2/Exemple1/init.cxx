#include "init.hxx"
#include <cstdlib>
#include <ctime>

void init(std::vector<double> & u)
{
   size_t i, n = u.size();

   for (i = 0; i<n; i++)
     u[i] = i*1.0/n;
}
