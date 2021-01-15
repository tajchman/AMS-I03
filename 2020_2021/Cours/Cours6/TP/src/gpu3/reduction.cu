/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
  Parallel reduction kernels
*/

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include "cuda_check.cuh"

__global__ void
reduce(double *g_odata, double *g_idata, size_t n)
{
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ double sdata[];
  
  size_t tid = threadIdx.x;
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  
  sdata[tid] = (i < n) ? g_idata[i] : 0;

  cg::sync(cta);

  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
      if (tid < s)
	      sdata[tid] += sdata[tid + s];

      cg::sync(cta);
    }
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

double reduction(
  double *d_out,
  double *d_in,
  size_t n, 
  int gridSize, int blockSize)
{ 
  double s;
  std::vector<double> s_block(gridSize);
  std::cout << std::endl;
  int smemSize = (blockSize <= 32)
    ? 2 * blockSize * sizeof(double)
    : blockSize * sizeof(double);
  
  reduce<<< gridSize, blockSize, smemSize >>>(d_out, d_in, n);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK_OP(cudaMemcpy(s_block.data(),
  			   d_out,
  			   gridSize * sizeof(double),
  			   cudaMemcpyDeviceToHost));
  
  s = 0.0;
  for (int i=0; i<gridSize; i++)
    s += s_block[i];
  return s;
}
