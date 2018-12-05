#include "add.hxx"

__global__ void vecInit(double *a, double *b, int n)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  double x;
  
  if (id >= n) return;

  x = 1.0 * id;
  a[id] = sin(x)*sin(x);
  b[id] = cos(x)*cos(x);
}

void init_GPU    (double **u,
		  double **v,
		  size_t n)
{
  int blockSize, gridSize;
  
  size_t bytes = n*sizeof(double);
  
  cudaMalloc(u, bytes);
  cudaMalloc(v, bytes);
   
  blockSize = 1024;
  
  gridSize = (int)ceil((float)n/blockSize);
  
  vecInit<<<gridSize, blockSize>>>(*u, *v, n);
  
}