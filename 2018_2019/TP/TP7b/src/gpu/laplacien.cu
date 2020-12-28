
#include "calcul.h"
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include "reduction.h"

__global__  void laplacienCuda(double * v, const double * u,
			      double dx, int n)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  
  if (i>0 && i < n-1 && j> 0 && j < n-1) {
    double L = 0.5/(dx*dx);
    
    v[i*n + j] =   -L * (4*u[i*n + j]
			- u[(i+1)*n + j] - u[(i-1)*n + j]
			- u[i*n + (j+1)] - u[i*n + (j-1)]);
  }
}

void laplacien(double * v, const double * u,
	       double dt, int n)
{
  dim3 blockSize(32,32);
  dim3 gridSize((n+blockSize.x)/blockSize.x, (n+blockSize.y)/blockSize.y);

  laplacienCuda<<<gridSize, blockSize>>>(v, u, dt, n);
}

