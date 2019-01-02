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
reduce(double *g_idata, double *g_odata, size_t n)
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

double reduce(size_t n,
              double *d_idata,
              double *d_odata,
	      int threads, int blocks)
{  
  double s;
  std::vector<double> s_block(blocks);
  
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  int smemSize = (threads <= 32)
    ? 2 * threads * sizeof(double)
    : threads * sizeof(double);
  
  reduce<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, n);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK_OP(cudaMemcpy(s_block.data(),
  			   d_odata,
  			   blocks * sizeof(double),
  			   cudaMemcpyDeviceToHost));
  
  s = 0.0;
  for (int i=0; i<blocks; i++)
    s += s_block[i];
  return s;
}
