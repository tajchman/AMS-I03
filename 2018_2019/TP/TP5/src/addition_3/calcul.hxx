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

class Calcul_GPU {
public:
  Calcul_GPU(std::size_t n);
  ~Calcul_GPU();
  
  void addition();
  double verification();

private:
  std::size_t m_n;
  double *d_u, *d_v, *d_w, *d_tmp;
  int gridSize, blockSize;
};

#endif

