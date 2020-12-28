#pragma once

#include <cuda.h>
#include <vector>

class VectorCUDA {

public:

  VectorCUDA(size_t n);
  ~VectorCUDA();

private:
  size_t m_n;
  double * c;
}