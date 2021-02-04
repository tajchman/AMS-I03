#include "calcul.h"
#include "cuda_check.cuh"
#include <cmath>
#include <cstdio>
#include <cuda.h>


__global__  void initCuda(double *u, int n)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  
  if (j > 0.2*n && i > 0.4*n && i < 0.6*n && j < 0.8*n)
    u[i*n+j] = 1.0;
  else
    u[i*n+j] = 0.0;
}

double * init(int n)
{
  double *u = alloue(n);

  dim3 blockSize(16, 16);
  dim3 gridSize((n + blockSize.x)/blockSize.x,
		(n + blockSize.y)/blockSize.y);

  initCuda<<<gridSize, blockSize>>>(u, n);
  checkKernel;
 
  return u;
}

__global__  void zeroCuda(double *u, int n)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < n)
    u[id] = 0.0;
}

double * zero(int n)
{
  double *u = alloue(n);

  int blockSize = 512;
  int gridSize = (n*n + blockSize)/blockSize;

  zeroCuda<<<gridSize, blockSize>>>(u, n*n);
  checkKernel;

  return u;
}


