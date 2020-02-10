#include "MatrixVectorCUDA.cuh"

VectorCUDA::VectorCUDA(size_t n) : m_n(n)
{
  cudaMalloc(&(c), n*sizeof(double));
}

VectorCUDA::~VectorCUDA()
{
  cudaFree(c);
}

MatrixCUDA::MatrixCUDA(size_t n, size_t m) : m_n(n), m_m(m)
{
  cudaMalloc(&(c), n*m*sizeof(double));
}

VectorCUDA::~VectorCUDA()
{
  cudaFree(c);
}

