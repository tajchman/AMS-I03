#include <vector>
#include <iostream>

int main(int argc, char **argv)
{
  int i, j;
  int n = argc > 1 ? strtol(argv[1], nullptr, 10) : 1000;

  // On cree 2 vecteurs de taille n initialises a 0
  std::vector<double> a(n, 0.0);
  std::vector<double> b(n, 0.0);
  
  for (i=1; i<n; i++) {
    a[i] = 0.5 + a[i-1];
    b[i] = b[i-1] - 0.5;
  }
  
  return 0;
}
