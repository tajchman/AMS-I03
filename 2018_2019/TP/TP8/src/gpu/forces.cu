#include "calcul.h"
#include <cmath>
#include <cstdio>
#include <cuda.h>

__global__  void forceCuda(double * f, const double * u, int n)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  
  if (i < n && j < n) {

    int
      i0 = n/10,
      i1 = n-n/10,
      j0 = n/2 - n/10,
      j1 = n/2 + n/10;
    double uu,  ff;
      
    uu = u[i*n+j];
      
    if (i>i0 && i<i1 && j>j0 && j<j1)
       ff = (20 - 20*uu*uu);
     else
       ff = 0.0;

    f[i*n+j] = ff;
  }
}

void calcul_forces(double * f, const double * u, int n)
{
  dim3 blockSize(16,16);
  dim3 gridSize((n+16)/16, (n+16)/16);

  forceCuda<<<gridSize, blockSize>>>(f, u, n);
}

