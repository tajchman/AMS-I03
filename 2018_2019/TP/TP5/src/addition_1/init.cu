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

void init_GPU(std::vector<double> &u,
	      std::vector<double> &v)
{
  double *d_u, *d_v;
  int blockSize, gridSize, n = u.size();
  
  size_t bytes = n*sizeof(double);
  
  cudaMalloc(&d_u, bytes);
  cudaMalloc(&d_v, bytes);
   
  blockSize = 1024;
  
  gridSize = (int)ceil((float)n/blockSize);
  
  vecInit<<<gridSize, blockSize>>>(d_u, d_v, n);
  
  cudaMemcpy(u.data(), d_u, bytes, cudaMemcpyDeviceToHost );
  cudaMemcpy(v.data(), d_v, bytes, cudaMemcpyDeviceToHost );
  
  cudaFree(d_u);
  cudaFree(d_v);
}