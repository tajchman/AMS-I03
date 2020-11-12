#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
int main() {
   size_t i, n = 1000000;
   std::vector<double> u(n), v(n);
	   
   std::srand(std::time(nullptr));
   for (i = 0; i<n; i++)
     u[i] = 2.0 + std::rand();
	     
   v[0] = u[0];
   for (i = 1; i<n-1; i++)
      v[i] = (u[i-1]+2*u[i]+u[i+1])/4;
   v[n-1] = u[n-1];    
   return 0;
}
