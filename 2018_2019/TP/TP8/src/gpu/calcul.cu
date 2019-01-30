
#include "calcul.h"
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include "reduction.h"

double * alloue(int n)
{
  double *d;
  size_t bytes = sizeof(double) * n * n;
  cudaMalloc(&d, bytes);
  return d;
}

double * alloue_work(int n)
{
  int nThreads = 512;
  int nBlocks = (n*n+nThreads)/nThreads;
  
  double * w;
  cudaMalloc(&w, sizeof(double) * nBlocks);
  return w;
}

void libere(double ** u)
{
  cudaFree(*u);
  *u = NULL;
}


double difference(const double * u, const double * v,
		  double * work, int n)
{
  double d;
  int nThreads = 512;
  int nBlocks = (n*n+512)/512;
  
  d = diff_reduce(n*n, u, v, work, nThreads, nBlocks);
 
  return d;
}
