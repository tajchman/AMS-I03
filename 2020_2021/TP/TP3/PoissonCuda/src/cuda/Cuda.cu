#include "Cuda.hxx"
#include "cuda_check.cuh"
#include "timer_id.hxx"

double * allocate(int n) {
    double *d;
    CUDA_CHECK_OP(cudaMalloc(&d, n*sizeof(double)));
    return d;
  }
  
  void deallocate(double *&d) {
    Timer & T = GetTimer(T_CopyId); T.start();
    CUDA_CHECK_OP(cudaFree(d));
    d = NULL;
    T.stop();
  }
  
  void copyDeviceToHost(double *h, double *d, int n)
  {
    Timer & T = GetTimer(T_CopyId); T.start();
    cudaMemcpy(h, d, n * sizeof(double), cudaMemcpyDeviceToHost);
    T.stop();
  }
  
  void copyHostToDevice(double *h, double *d, int n)
  {
    Timer & T = GetTimer(T_CopyId); T.start();
    cudaMemcpy(h, d, n * sizeof(double), cudaMemcpyHostToDevice);
    T.stop();
  }
  
  void copyDeviceToDevice(double *d_out, double *d_in, int n)
  {
    Timer & T = GetTimer(T_CopyId); T.start();
    cudaMemcpy(d_out, d_in, n * sizeof(double), cudaMemcpyDeviceToDevice);
    T.stop();
  }
  
  
