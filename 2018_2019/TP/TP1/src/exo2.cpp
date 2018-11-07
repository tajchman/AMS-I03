#include <vector>
#include <iostream>

int main(int argc, char **argv)
{
  int i, j;
  int n = argc > 1 ? strtol(argv[1], nullptr, 10) : 1000;

  // On cree 2 vecteurs de taille n initialises a 0
  std::vector<double> a(n, 1.0);
  std::vector<double> b(n, 2.0);
  
  double sum;
  for (const double & x : a)
    sum += x;
  for (const double & y : b)
    sum += y;
  
  return 0;
}
