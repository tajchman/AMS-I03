#include <iostream>
#include "timer.hxx"
#include "cuda_check.cuh"

__device__ double my_rand()
{
  long int hi = seed / q;
  long int lo = seed % q;
  long int test = a * lo - r * hi;
  if(test > 0)
    seed = test;
  else	seed = test + m;
  return (double) seed/m;
}

__global__ void tests(short *u, long n)
{
  double x, y;
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (id >= n) return;

  x = my_rand();
  y = my_rand();

  u[id] = x*x + y*y < 1 ? 1 : 0;
}

double Calcul_Pi(std::size_t n)
{
  short * d_u;
  std::size_t bytes = n*sizeof(int);
  CUDA_CHECK_OP(cudaMalloc(&d_u, bytes));
   
  int blockSize = 512;
  int gridSize = (int)ceil((double)n/blockSize);
  
  tests<<<gridSize, blockSize>>>(d_u, n);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK_OP(cudaFree(d_u));
  return 0.0;
}

