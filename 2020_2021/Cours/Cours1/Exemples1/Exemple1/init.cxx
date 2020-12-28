#include "init.hxx"
#include <cstdlib>
#include <ctime>

void init(std::vector<double> & u)
{
   size_t i, n = u.size();

   std::srand(std::time(nullptr));
   for (i = 0; i<n; i++)
     u[i] = 2.0 + std::rand();
}
