
#include "calcul.h"
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include "cuda_check.cuh"
#include "reduction.h"

double * alloue(int n)
{
  double *d;
  size_t bytes = sizeof(double) * n * n;
  checkCuda(cudaMalloc(&d, bytes));
  return d;
}

double * alloue_work(int n)
{
  int nThreads = 512;
  int nBlocks = (n*n+nThreads)/nThreads;
  
  double * w;
  checkCuda(cudaMalloc(&w, sizeof(double) * nBlocks));
  return w;
}

void libere(double ** u)
{
  checkCuda(cudaFree(*u));
  *u = NULL;
}


double difference(const double * u, const double * v,
		  double * work, int n)
{
  double d;
  int nThreads = 512;
  int nBlocks = (n*n+nThreads)/nThreads;
  
  d = diff_reduce(n*n, u, v, work, nThreads, nBlocks);
 
  return d;
}
