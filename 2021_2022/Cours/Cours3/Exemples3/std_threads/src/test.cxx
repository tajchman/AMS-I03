#include <cstdlib>
#include <iostream>
#include <vector>

void wait(const char *message) 
{
  do 
  {
    std::cout << '\n' << message << "....";
  } while (std::cin.get() != '\n');
}

int main()
{
   int n = 1000000000;

   double *v = new double[n];

   wait("apres la construction de v");

   for (int i=0; i<n/2; i++)
     v[i] = 1.0;

   wait("apres l'initialisation de la 1ere moitie)");

   for (int i=n/2; i<n; i++)
     v[i] = 1.0;

   wait("apres l'initialisation de la 2nde moitie)");
   
   delete [] v;
   wait("apres destruction v");

   return 0;
}
