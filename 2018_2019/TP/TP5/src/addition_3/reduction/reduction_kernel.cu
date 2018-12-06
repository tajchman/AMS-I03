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
#include <cooperative_groups.h>
#include "cuda_check.cuh"

namespace cg = cooperative_groups;

struct SharedMemory
{
  __device__ inline operator       double *()
  {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const
  {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};


template <unsigned int blockSize>
__global__ void reduce(double *g_idata, double *g_odata, unsigned int n)
{
  cg::thread_block cta = cg::this_thread_block();
  double *sdata = SharedMemory();

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  unsigned int i0 = i;
  double mySum = 0;
  while (i < n) {
    if (i0 < 10) printf ("i_data %d = %g\n", i, g_idata[i]);
    mySum += g_idata[i];
    
    if (i + blockSize < n)
      mySum += g_idata[i+blockSize];
    
    i += gridSize;
    }
  f (i0 < 10) printf("%d : mySum : %f\n", i0, mySum);
//
//  // each thread puts its local sum into shared memory
//  sdata[tid] = mySum;
//  cg::sync(cta);
//
//
//  // do reduction in shared mem
//  if ((blockSize >= 512) && (tid < 256))
//    {
//      sdata[tid] = mySum = mySum + sdata[tid + 256];
//    }
//
//  cg::sync(cta);
//
//  if ((blockSize >= 256) &&(tid < 128))
//    {
//      sdata[tid] = mySum = mySum + sdata[tid + 128];
//    }
//
//  cg::sync(cta);
//
//  if ((blockSize >= 128) && (tid <  64))
//    {
//      sdata[tid] = mySum = mySum + sdata[tid +  64];
//    }
//
//  cg::sync(cta);
//
//#if (__CUDA_ARCH__ >= 300 )
//  if ( tid < 32 )
//    {
//      cg::coalesced_group active = cg::coalesced_threads();
//
//      // Fetch final intermediate sum from 2nd warp
//      if (blockSize >=  64) mySum += sdata[tid + 32];
//      // Reduce final warp using shuffle
//      for (int offset = warpSize/2; offset > 0; offset /= 2) 
//        {
//          mySum += active.shfl_down(mySum, offset);
//        }
//    }
//#else
//  // fully unroll reduction within a single warp
//  if ((blockSize >=  64) && (tid < 32))
//    {
//      sdata[tid] = mySum = mySum + sdata[tid + 32];
//    }
//
//  cg::sync(cta);
//
//  if ((blockSize >=  32) && (tid < 16))
//    {
//      sdata[tid] = mySum = mySum + sdata[tid + 16];
//    }
//
//  cg::sync(cta);
//
//  if ((blockSize >=  16) && (tid <  8))
//    {
//      sdata[tid] = mySum = mySum + sdata[tid +  8];
//    }
//
//  cg::sync(cta);
//
//  if ((blockSize >=   8) && (tid <  4))
//    {
//      sdata[tid] = mySum = mySum + sdata[tid +  4];
//    }
//
//  cg::sync(cta);
//
//  if ((blockSize >=   4) && (tid <  2))
//    {
//      sdata[tid] = mySum = mySum + sdata[tid +  2];
//    }
//
//  cg::sync(cta);
//
//  if ((blockSize >=   2) && ( tid <  1))
//    {
//      sdata[tid] = mySum = mySum + sdata[tid +  1];
//    }
//
//  cg::sync(cta);
//#endif
//
//  if (tid == 0) g_odata[blockIdx.x] = mySum;
}


double reduce(int size,
              int threads,
              int blocks,
              double *d_idata,
              double *d_odata)
{
  double s;
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  
  int smemSize = (threads <= 32)
    ? 2 * threads * sizeof(double)
    : threads * sizeof(double);
  
  switch (threads)
    {
    case 512:
      reduce< 512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
      break;
      
    case 256:
      reduce< 256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
      break;
      
    case 128:
      reduce< 128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
      break;
      
    case 64:
      reduce<  64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
      break;
      
    case 32:
      reduce<  32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
      break;
      
    case 16:
      reduce<  16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
      break;
      
    case  8:
      reduce<   8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
      break;
      
    case  4:
      reduce<   4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
      break;
      
    case  2:
      reduce<   2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
      break;
      
    case  1:
      reduce<   1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
      break;
    }
  CUDA_CHECK_KERNEL();
  //CUDA_CHECK_OP(cudaMemcpy(&s, d_odata, sizeof(double), cudaMemcpyDeviceToHost));
  return s;
}
