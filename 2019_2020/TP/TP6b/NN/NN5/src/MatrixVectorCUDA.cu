#include "MatrixVector.h"

__global__
void _zero(int n, double *x)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      x[i] = 0;
}

__global__
void _radd(int n, double *x, double *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      x[i] += y[i];
}

__global__
void _add(int n, double *x, double *y, double *z)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      x[i] = y[i] + z[i];
}

void Vector::resize(size_t n)
{
  m_n = n;
  if (m_c) cudaFree(m_c);
  cudaMalloc(&(m_c), m_n*sizeof(double));
}

Vector::~Vector()
{
  if (m_c) cudaFree(m_c);
}

Vector::Vector(const std::vector<double> & v)
{
  m_n = v.size();
  cudaMemcpy(m_c, v.data(), m_n * sizeof(double), cudaMemcpyHostToDevice); 
}

void Vector::operator=(const std::vector<double> & v)
{
  if (m_n != v.size())
    throw "Vector::operator= dimensions";
  
  cudaMemcpy(m_c, v.data(), m_n * sizeof(double), cudaMemcpyHostToDevice); 
}

void Vector::zero()
{
  int blockSize = 256;
  int numBlocks = (m_n + blockSize - 1) / blockSize;
  _zero<<<numBlocks, blockSize>>>((int) m_n, m_c);
}

void Matrix::resize(size_t n, size_t m)
{
  m_n = n;
  m_m = m;
  if (m_c) cudaFree(m_c);
  cudaMalloc(&(m_c), m_n*m_m*sizeof(double));
}

Matrix::~Matrix()
{
  if (m_c) cudaFree(m_c);
}

