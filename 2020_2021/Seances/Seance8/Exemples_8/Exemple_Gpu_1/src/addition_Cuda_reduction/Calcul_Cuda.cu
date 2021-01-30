#include <iostream>
#include "Calcul_Cuda.hxx"
#include "timer.hxx"
#include "reduction.h"
#include "cuda_check.cuh"

Calcul_Cuda::Calcul_Cuda(int m) : n(m)
{
  Timer & T = GetTimer(T_AllocId); T.start();
  
  int bytes = n*sizeof(double);
  CUDA_CHECK_OP(cudaMalloc(&d_u, bytes));
  CUDA_CHECK_OP(cudaMalloc(&d_v, bytes));
  CUDA_CHECK_OP(cudaMalloc(&d_w, bytes));
  CUDA_CHECK_OP(cudaMalloc(&d_tmp, bytes));

  T.stop();
}

Calcul_Cuda::~Calcul_Cuda()
{
  Timer & T = GetTimer(T_FreeId); T.start();

  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_tmp);

  T.stop();
}

__global__ void vecInit(double *a, double *b, int n)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  double x;
  
  if (id >= n) return;

  x = 1.0 * id;
  a[id] = sin(x)*sin(x);
  b[id] = cos(x)*cos(x);
}

void Calcul_Cuda::init()
{
  Timer & T = GetTimer(T_InitId); T.start();

  blockSize = 512;
  gridSize = (unsigned int) ceil((double)n/blockSize);
  
  vecInit<<<gridSize, blockSize>>>(d_u, d_v, n);
  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();

  T.stop();
}

__global__ void vecAdd(double *c, double *a, double *b, int n)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (id < n)
    c[id] = a[id] + b[id];
}

void Calcul_Cuda::addition()
{
  Timer & T = GetTimer(T_AddId); T.start();
  
  vecAdd<<<gridSize, blockSize>>>(d_w, d_u, d_v, n);
  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();

  T.stop();
}

double Calcul_Cuda::verification()
{
  Timer & T = GetTimer(T_VerifId);
  T.start();
  
  double s;
  s = reduce(n, d_w, d_tmp, blockSize, gridSize);  
  s = s/n - 1.0;
  
  T.stop();

  return s;
}

