#include "init.hxx"
#include "calcul.hxx"
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  size_t  n = 400000000;
  std::vector<double> v(n);
  int nThreads = argc > 1 ? strtol(argv[1], NULL, 10) : 1;
  int offset = argc > 2 ? strtol(argv[2], NULL, 10) : 1;

  std::cout << "Taille vecteur : " << n << std::endl;
  std::cout << "Threads        : " << nThreads << std::endl;
  std::cout << "Offset         : " << offset << std::endl;

  init(v);

  std::cout << "\nCalcul sequentiel  : " << std::endl;
  double v0 = somme0(v);

  std::cout << "\nCalcul parallele (offset = 1)  : " << std::endl;
  double v1 = somme1(v, nThreads);
  //std::cout << "  difference = " << (v1 - v0)/v0 << std::endl;

  std::cout << "\nCalcul parallele (offset = " << offset << ")  : " << std::endl;
  double v2 = somme2(v, nThreads, offset);
  //std::cout << "  difference = " << (v2 - v0)/v0 << std::endl;

  return 0;
 }
