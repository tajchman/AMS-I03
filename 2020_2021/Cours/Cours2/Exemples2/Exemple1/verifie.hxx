#include <iostream>
#include <vector>

inline void verifie(const std::vector<double> & v1, 
             const std::vector<double> & v2)
{
   std::cout << "verification ";

   double diff = 0;
   int i, n = v1.size();
   for (i=0; i<n; i++)
      diff += std::abs(v1[i] - v2[i]);
   if (diff < 1e-15)
      std::cout << "oui";
   else
      std::cout << "non";
   std::cout << std::endl<< std::endl;
}

