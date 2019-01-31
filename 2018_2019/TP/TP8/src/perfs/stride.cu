/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <stdio.h>
#include <assert.h>
#include "timerGPU.cuh"

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.

#include "cuda_check.cuh"

FILE * out_offset, * out_stride;

template <typename T>
__global__ void offset(T* a, int s)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x + s;
  a[i] = a[i] + 1;
}

template <typename T>
__global__ void stride(T* a, int s)
{
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * s;
  a[i] = a[i] + 1;
}

template <typename T>
void runTest(int deviceId, int nMB)
{
  int blockSize = 256;

  T *d_a;    
  int n = nMB*1024*1024/sizeof(T);

  checkCuda( cudaMalloc(&d_a, n * 33 * sizeof(T)) );

  offset<<<n/blockSize, blockSize>>>(d_a, 0); // warm up

  for (int i = 0; i <= 32; i++) {
    checkCuda( cudaMemset(d_a, 0, n * sizeof(T)) );

    TimerGPU Tmp;
    Tmp.start();
    
    offset<<<n/blockSize, blockSize>>>(d_a, i);

    Tmp.stop();
    
    fprintf(out_offset, "%d %e\n", i, Tmp.elapsed());
  }

  printf("\n\n");
  stride<<<n/blockSize, blockSize>>>(d_a, 1); // warm up
  for (int i = 1; i <= 32; i++) {
    checkCuda( cudaMemset(d_a, 0, n * sizeof(T)) );

    TimerGPU Tmp;
    Tmp.start();
    
    stride<<<n/blockSize, blockSize>>>(d_a, i);

    Tmp.stop();

    fprintf(out_stride, "%d %g\n", i, Tmp.elapsed());
  }

  cudaFree(d_a);
}

int main(int argc, char **argv)
{
  int nMB = 4;
  int deviceId = 0;
  bool bFp64 = false;

  for (int i = 1; i < argc; i++) {    
    if (!strncmp(argv[i], "dev=", 4))
      deviceId = atoi((char*)(&argv[i][4]));
    else if (!strcmp(argv[i], "fp64"))
      bFp64 = true;
  }
  
  cudaDeviceProp prop;
  
  checkCuda( cudaSetDevice(deviceId));
  checkCuda( cudaGetDeviceProperties(&prop, deviceId) );
  printf("Device: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", nMB);
  
  printf("%s Precision\n", bFp64 ? "Double" : "Single");

  out_offset = fopen("offset.txt", "w");
  out_stride = fopen("stride.txt", "w");

  if (bFp64) runTest<double>(deviceId, nMB);
  else       runTest<float>(deviceId, nMB);

  fclose(out_offset);
  fclose(out_stride);
}
