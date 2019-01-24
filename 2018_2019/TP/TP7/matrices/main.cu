#include <stdio.h>
#include "sMatrice.h"

__global__ void multGPU(const double *U, const double *V, double *W,
			int n, int p, int m)
{
  int i, j, k;
  i = blockIdx.x*blockDim.x+threadIdx.x;
  j = blockIdx.y*blockDim.y+threadIdx.y;
  
  if ((i < n) && (j < m)) {
    double s = 0.0;
    for (k=0; k<p; k++) {
      s += U[i*p + k] * V[k*m + j];
    }
    //printf("%d,%d  s = %g\n", i, j, s);
    W[i*m + j] = s;
  }
}

void mult(const sMatrice &h_U, const sMatrice &h_V, sMatrice &h_W)
{
  double * d_U, * d_V, * d_W;

  size_t bytes;

  bytes = h_U.n() * h_U.m() * sizeof(double);
  cudaMalloc(&(d_U), bytes);
  cudaMemcpy(d_U, h_U.data(), bytes, cudaMemcpyHostToDevice);
		
  bytes = h_V.n() * h_V.m() * sizeof(double);
  cudaMalloc(&(d_V), bytes);
  cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice);
  
  bytes = h_W.n() * h_W.m() * sizeof(double);
  cudaMalloc(&(d_W), bytes);
  
  dim3 blockSize(16, 16);
  dim3 gridSize((h_W.n() + blockSize.x)/blockSize.x,
		(h_W.m() + blockSize.y)/blockSize.y); 
  multGPU<<<gridSize, blockSize>>>(d_U, d_V, d_W,
				   h_U.n(), h_V.n(), h_W.m());
		
  bytes = h_W.n() * h_W.m() * sizeof(double);
  cudaMemcpy(h_W.data(), d_W, bytes, cudaMemcpyDeviceToHost);
  
  cudaFree(d_W);
  cudaFree(d_V);
  cudaFree(d_U);
}

int main()
{
  int n = 5, m = 4, p = 10;

  sMatrice A(n, p, "A"), B(p, m, "B"), C(n, m, "C");

  int i,j;
  for (i=0; i<n; i++)
    for (j=0; j<p; j++)
      A(i,j) = i*100 + j;
  
  for (i=0; i<p; i++)
    for (j=0; j<m; j++)
      B(i,j) = i*100 + j;

  mult(A, B, C);

  std::cout << A << B << C;
  return 0;
}
