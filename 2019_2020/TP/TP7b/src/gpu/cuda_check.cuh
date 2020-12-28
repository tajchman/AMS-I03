#ifndef __CUDA_CHECK__
#define __CUDA_CHECK__

#include <iostream>
#include <cstdlib>

#define checkCuda(X) \
{ \
  cudaError_t result = X; \
  if (result != cudaSuccess) { \
    std::cerr << __FILE__ << "(" << __LINE__ << ") : CUDA Runtime Error: " \
	      << cudaGetErrorString(result) \
	      << std::endl; \
    std::exit(-1); \
  } \
}

#define checkKernel \
  { \
  cudaDeviceSynchronize(); \
  checkCuda(cudaGetLastError()); \
  }
#endif
