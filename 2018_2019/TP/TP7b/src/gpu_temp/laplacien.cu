
#include "calcul.h"
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include "reduction.h"

const int xTileSize = 1; 
const int yTileSize = 8;
 
__global__ void d2u_dx2(float *u, float *d2u, int n)
{
  __shared__ float s_u[xTileSize][n]; 
  //Global indices
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int si = threadIdx.x; 
  int sj = threadIdx.y; 

  int globalIdx = j * nx + i;

  s_phi[sj][si] = u[globalIdx];

  __syncthreads();

  if( i > 2 && i < (nx - 3) )
    {
      d2u[globalIdx] =
	( ax_c * ( s_phi[sj][si+1] + s_phi[sj][si-1] )
	  + bx_c * ( s_phi[sj][si+2] + s_phi[sj][si-2] )
	  + cx_c * ( s_phi[sj][si+3] + s_phi[sj][si-3] )
	  + dx_c * ( s_phi[sj][si]));
    }
}




// 2nd order derivative with respect to y
__global__ void d2u_dy2(float *u, float *d2u)
{
  __shared__ float s_u[ny][yTileSize];
  //Global and shared x indecies
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int si = threadIdx.x;

  for (int j = threadIdx.y; j < ny; j += blockDim.y)
    {
      int globalIdx =  j * nx + i;
      int sj = j;
      s_u[sj][si] = u[globalIdx];
    }
  
  __syncthreads();

  for (int j = threadIdx.y; j < ny; j += blockDim.y)
    {
      int globalIdx = j*nx + i;
      int sj = j;
      if( j > 2 && j < ny - 3)
	{
	  d2u[globalIdx] =
	    ( ay_c * ( s_phi[sj+1][si] + s_phi[sj-1][si] )
	      + by_c * ( s_phi[sj+2][si] + s_phi[sj-2][si] )
	      + cy_c * ( s_phi[sj+3][si] + s_phi[sj-3][si] )
	      + dy_c * ( s_phi[sj][si] ) );
	}
    }
}

__global__  void laplacienCuda(double * v, const double * u,
			      double dx, int n)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  
  if (i>0 && i < n-1 && j> 0 && j < n-1) {
    double L = 0.5/(dx*dx);
    double vv;

    vv = 4*u[i*n + j];
    vv -= u[(i+1)*n + j]
    
    v[i*n + j] =   -L * (4*u[i*n + j]
			- u[(i+1)*n + j] - u[(i-1)*n + j]
			- u[i*n + (j+1)] - u[i*n + (j-1)]);
    v[i*n + j] =   -L * (4*u[i*n + j]
			- u[(i+1)*n + j] - u[(i-1)*n + j]
			- u[i*n + (j+1)] - u[i*n + (j-1)]);
  }
}

void laplacien(double * v, const double * u,
	       double dt, int n)
{
  dim3 blockSize(16,16);
  dim3 gridSize((n+16)/16, (n+16)/16);

  laplacienCuda<<<gridSize, blockSize>>>(v, u, dt, n);
}

