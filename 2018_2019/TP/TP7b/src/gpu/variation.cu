#include "calcul.h"
#include <cuda.h>

__global__
void variationCuda(double * u_next, const double * u_current,
		   const double * u_diffuse, const double * forces,
		   double dt, int n)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  
  if (i > 0 && i < n && j > 0 && j < n) {

    u_next[i*n+j] = u_current[i*n+j]
      + dt * (u_diffuse[i*n+j] + forces[i*n+j]);
  }
}

void variation    (double * u_next, const double * u_current,
                   const double * u_diffuse, const double * forces,
                   double dt, int n)
{
  dim3 blockSize(16,16);
  dim3 gridSize((n+blockSize.x)/blockSize.x,
		(n+blockSize.y)/blockSize.y);

  variationCuda<<<gridSize, blockSize>>>(u_next, u_current,
					 u_diffuse, forces,
					 dt, n);
}
