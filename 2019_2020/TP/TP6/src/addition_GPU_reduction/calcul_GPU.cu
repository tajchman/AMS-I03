#include <iostream>
#include "calcul.hxx"
#include "timer.hxx"
#include "reduction.h"
#include "cuda_check.cuh"

__global__ void vecInit(double *a, double *b, int n)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  double x;
  
  if (id >= n) return;

  x = 1.0 * id;
  a[id] = sin(x)*sin(x);
  b[id] = cos(x)*cos(x);
}

__global__ void vecAdd(double *c, double *a, double *b, int n)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (id < n)
    c[id] = a[id] + b[id];
}


Calcul_GPU::Calcul_GPU(std::size_t n) : m_n(n)
{
  Timer T1; T1.start();
  
  std::size_t bytes = m_n*sizeof(double);
  CUDA_CHECK_OP(cudaMalloc(&d_u, bytes));
  CUDA_CHECK_OP(cudaMalloc(&d_v, bytes));
  CUDA_CHECK_OP(cudaMalloc(&d_w, bytes));
  CUDA_CHECK_OP(cudaMalloc(&d_tmp, bytes));
    
  T1.stop();
  std::cerr << "\t\ttemps init 1 : " << T1.elapsed() << std::endl;
  Timer T2; T2.start();
  
  blockSize = 512;
  gridSize = (int)ceil((double)n/blockSize);
  
  vecInit<<<gridSize, blockSize>>>(d_u, d_v, n);
  CUDA_CHECK_KERNEL();

  T2.stop();
  std::cerr << "\t\ttemps init 2 : " << T2.elapsed() << std::endl;
}

Calcul_GPU::~Calcul_GPU()
{
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_tmp);
}

void Calcul_GPU::addition()
{
  Timer T; T.start();
  
  vecAdd<<<gridSize, blockSize>>>(d_w, d_u, d_v, m_n);
  CUDA_CHECK_KERNEL();
  
  T.stop();
  std::cerr << "\t\ttemps add.   : " << T.elapsed() << std::endl;
}

double Calcul_GPU::verification()
{
  Timer T; T.start();
  
  double s;
  s = reduce(m_n, d_w, d_tmp, blockSize, gridSize);  
  s = s/m_n - 1.0;
  
  T.stop();
  std::cerr << "\t\ttemps verif. : " << T.elapsed() << std::endl;

  return s;
}

