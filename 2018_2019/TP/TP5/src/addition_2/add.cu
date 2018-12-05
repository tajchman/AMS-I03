#include "add.hxx"

__global__ void vecAdd(double *c, double *a, double *b, int n)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (id < n)
    c[id] = a[id] + b[id];
}

void addition_GPU(double **w,
		  double *u,
		  double *v,
		  size_t n)
{
  double *d_u, *d_v, *d_w;
  int blockSize, gridSize;
  
  size_t bytes = n*sizeof(double);
  
  cudaMalloc(&d_u, bytes);
  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_w, bytes);
  
  cudaMemcpy( d_u, u.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( d_v, v.data(), bytes, cudaMemcpyHostToDevice);
 
  blockSize = 1024;
  
  gridSize = (int)ceil((float)n/blockSize);
  
  vecAdd<<<gridSize, blockSize>>>(d_w, d_u, d_v, n);
  
  cudaMemcpy(w.data(), d_w, bytes, cudaMemcpyDeviceToHost );
  
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
}