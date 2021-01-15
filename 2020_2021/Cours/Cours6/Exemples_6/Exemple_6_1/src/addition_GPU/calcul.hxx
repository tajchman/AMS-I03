#ifndef _CALCUL_HXX
#define _CALCUL_HXX

#include <vector>

class Calcul_GPU {
public:
  Calcul_GPU(std::size_t n);
  ~Calcul_GPU();
  
  void addition();
  double verification();

private:
  std::size_t m_n;
  double *d_u, *d_v, *d_w;
  unsigned int blockSize, gridSize;
};

#endif

