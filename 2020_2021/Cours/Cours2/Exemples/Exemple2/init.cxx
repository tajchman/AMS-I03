#include "init.hxx"
#include <cstdlib>
#include <ctime>

void init(std::vector<double> & v)
{
   size_t i, j, n = v.size();

   std::srand(std::time(nullptr));
   for (i = 0; i<n; i++)
      v[i] = 0.00001 * (2.0 + std::rand());
}
