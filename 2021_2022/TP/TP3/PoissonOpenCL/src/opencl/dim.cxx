#include "dim.hxx"
#include "timer_id.hxx"

__constant int d_n[3];
__constant double d_xmin[3];
__constant double d_dx[3];
__constant double d_lambda[3];


void setDims(const int *h_n, 
             const double *h_xmin, 
             const double *h_dx, 
             const double *h_lambda)
{
    Timer & T = GetTimer(T_CopyId); T.start();
    d_n = {h_n[0], h_n[1], h_n[2]};
    CUDA_CHECK_OP(cudaMemcpyToSymbol(d_xmin, h_xmin, 3 * sizeof(double)));
    CUDA_CHECK_OP(cudaMemcpyToSymbol(d_dx, h_dx, 3 * sizeof(double)));
    CUDA_CHECK_OP(cudaMemcpyToSymbol(d_lambda, h_lambda, 3 * sizeof(double)));
    cudaDeviceSynchronize();
    T.stop();
  
//    symbol<<<1,1>>>();  
}