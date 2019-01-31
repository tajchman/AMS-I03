#ifndef __CUDA_CHECK__
#define __CUDA_CHECK__

#include <iostream>
#include <cstdlib>

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result)
	      << std::endl;;
    std::exit(-1);
  }
#endif
  return result;
}

#endif
