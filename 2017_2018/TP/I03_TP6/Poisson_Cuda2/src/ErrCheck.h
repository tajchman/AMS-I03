#ifndef __ERRCHECK_H
#define __ERRCHECK_H

#include <cuda.h>

#define CHECK_CUDA(ans) { CUDA_ASSERT((ans), __FILE__, __LINE__); }
inline void CUDA_ASSERT(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

