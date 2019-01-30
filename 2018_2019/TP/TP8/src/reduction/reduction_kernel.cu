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
#include <math.h>

namespace cg = cooperative_groups;

__global__ void
diff_reduce(const double *g_idata1, const double *g_idata2,
	    double *g_odata, size_t n)
{
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ double sdata[];
  
  size_t tid = threadIdx.x;
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? fabs(g_idata1[i] - g_idata2[i]) : 0;

  cg::sync(cta);

  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
      if (tid < s)
	  sdata[tid] += sdata[tid + s];

      cg::sync(cta);
    }
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

double * work_reduce(int n)
{
  double * w;
  cudaMalloc(&w, sizeof(double) * n);
  return w;
}

double diff_reduce(size_t n,
		   const double *d_idata1,
		   const double *d_idata2,
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
  
  diff_reduce<<< dimGrid, dimBlock, smemSize >>>(d_idata1, d_idata2,
						 d_odata, n);
  cudaMemcpy(s_block.data(),
	     d_odata,
	     blocks * sizeof(double),
	     cudaMemcpyDeviceToHost);

  s = 0.0;
  for (int i=0; i<blocks; i++)
    s += s_block[i];
  return s;
}
