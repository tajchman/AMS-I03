
#include "calcul.h"
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include "reduction.h"

__global__  void initCuda(double *u, int n)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < n)
    u[id] = 0.0;
}


double * init(int n)
{
  double *d = alloue(n);

  int blockSize = 512;
  int gridSize = (n*n + 512)/512;

  initCuda<<<gridSize, blockSize>>>(d, n);
  
  return d;
}

double * alloue(int n)
{
  double *d;
  size_t bytes = sizeof(double) * n * n;
  cudaMalloc(&d, bytes);
  return d;
}

void libere(double ** u)
{
  cudaFree(*u);
  *u = NULL;
}

__device__  void laplacienCuda(double * v, const double * u,
			       int i, int j, int n, double lambda)
{
    v[i*n + j] = u[i*n + j]
      - lambda * (4*u[i*n + j]
		  - u[(i+1)*n + j] - u[(i-1)*n + j]
		  - u[i*n + (j+1)] - u[i*n + (j-1)]);
}

__device__  void forceCuda(double * f, double u, int i, int j, int n)
{
    if (i==3*n/4 && j>n/4 && j<3*n/4)
      *f = (0.25 - u*u);
    else if (i==n/4 && j>n/4 && j<3*n/4)
      *f = -(0.25 - u*u);
    else
      *f = 0.0;
}

__global__  void iterationCuda(double * v, const double * u, double dt, int n)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  
  if (i>0 && i < n-1 && j> 0 && j < n-1) {
    double f;
    double lambda = 0.25;
    
    laplacienCuda(v, u, i, j, n, lambda);
    forceCuda(&f, u[i*n+j], i, j, n);
    printf("i = %03d j = %03d f = %16.10g\n", i, j, f);
    v[i*n + j] += f*dt;
  }
}


void iteration(double * v, const double * u, double dt, int n)
{
  dim3 blockSize(16,16);
  dim3 gridSize((n+16)/16, (n+16)/16);

  iterationCuda<<<gridSize, blockSize>>>(v, u, dt, n);
}


double difference(const double * u, const double * v, int n)
{
  double *w, d;
  int nThreads = 512;
  int nBlocks = (n*n+512)/512;
  
  w = work_reduce(nBlocks);
  d = diff_reduce(n*n, u, v, w, nThreads, nBlocks);
 
  return d;
}

void save(const char *name, const double *u, int n)
{
  int i,j;
  
  std::size_t bytes = n*n*sizeof(double);
  std::vector<double> w(n*n);
  cudaMemcpy(w.data(), u, bytes, cudaMemcpyDeviceToHost);

  std::string s = "gpu_";
  s += name;
  
  FILE * f = fopen(s.c_str(), "w");
  
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++)
      fprintf(f, "%g %g %g\n", i*1.0/n, j*1.0/n, w[i*n+j]);
    fprintf(f,"\n");
  }
  fprintf(f,"\n");

  fclose(f);
}
