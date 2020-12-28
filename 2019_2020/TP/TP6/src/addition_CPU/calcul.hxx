#ifndef _CALCUL_HXX
#define _CALCUL_HXX

#include <vector>

class Calcul_CPU {
public:
  Calcul_CPU(std::size_t n);
  
  void addition();
  double verification();

private:
  std::vector<double> h_u, h_v, h_w;
};

#endif

