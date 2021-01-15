#include <iostream>
#include "timerGPU.hxx"
#include <cuda.h>
#include <curand_kernel.h>
#include "cuda_check.cuh"

const long ITERATIONS = 10000L;
const int BLOCKSIZE = 512;

__global__ void tests(long * totals)
{
  __shared__ long counter[BLOCKSIZE];

  float x, y;
  int thread_id = threadIdx.x,
    block_id = blockIdx.x,
    global_id = block_id * blockDim.x + thread_id;
  
  curandState_t rng;
  curand_init(clock64(), global_id, 0, &rng);
  
  int c = 0;
  for (int i = 0; i < ITERATIONS; i++) {
    x = curand_uniform(&rng);
    y = curand_uniform(&rng); 
    c += 1 - int(x * x + y * y); 
  }

  counter[thread_id] = c;
  
  __syncthreads();
  
  if (threadIdx.x == 0) {
    totals[block_id] = 0;
    for (int i = 0; i < BLOCKSIZE; i++) {
      totals[block_id] += counter[i];
    }
  }
}

double Calcul_Pi(std::size_t n)
{  
  int numDev;
  cudaGetDeviceCount(&numDev);
  if (numDev < 1) {
    std::cerr << "CUDA device missing!\n";
    return 0.0;
  }

  int gridSize = (int)ceil((double)n/BLOCKSIZE);

  long * h_compte, * d_compte;
  h_compte = new long[gridSize];
  cudaMalloc(&d_compte, sizeof(long) * gridSize);

  tests<<<gridSize, BLOCKSIZE>>>(d_compte);
  CUDA_CHECK_KERNEL();

  cudaMemcpy(h_compte, d_compte, sizeof(long) * gridSize,
	     cudaMemcpyDeviceToHost);
  
  CUDA_CHECK_OP(cudaFree(d_compte));

  long total = 0;
  for (int i = 0; i < gridSize; i++) {
    total += h_compte[i];
  }
  long ntests = BLOCKSIZE * gridSize * ITERATIONS;

  return 4.0 * (double)total/(double)ntests;
}

