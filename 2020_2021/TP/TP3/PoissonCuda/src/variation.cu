

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include "cuda_check.cuh"

__global__ void
reduce(const double *u, const double *v, double *partialDiff, int n)
{
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ double sdata[];
  
  int tid = threadIdx.x;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  
  sdata[tid] = (i < n) ? abs(u[i] - v[i]) : 0;

  cg::sync(cta);

  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
      if (tid < s)
        sdata[tid] += sdata[tid + s];

      cg::sync(cta);
    }
  if (tid == 0) partialDiff[blockIdx.x] = sdata[0];
}

double variationCuda(double *u, double *v, double * &d_partialSums, int n)
{  
  int dimBlock = 512;
  int dimGrid = ceil(n/double(dimBlock));
  int nbytes = dimBlock * sizeof(double);

  int smemSize = nbytes;
  if (d_partialSums == NULL)
     CUDA_CHECK_OP(cudaMalloc(&d_partialSums, nbytes));

  reduce<<< dimGrid, dimBlock, smemSize >>>(u, v, d_partialSums, n);
  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();
  
  std::vector<double> h_partialSums(dimBlock);
  for (int i=0; i<dimBlock; i++)
    h_partialSums[i] = 0.0;

  CUDA_CHECK_OP(cudaMemcpy(h_partialSums.data(), d_partialSums,
                           nbytes, cudaMemcpyDeviceToHost));
  
  double s = 0.0;
  for (int i=0; i<dimBlock; i++)
    s += h_partialSums[i];
  return s;
}
