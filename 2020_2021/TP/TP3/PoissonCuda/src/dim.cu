#include "dim.cuh"
#include "dim.hxx"
#include "timer_id.hxx"

__constant__ int n[3];
__constant__ double xmin[3];
__constant__ double dx[3];
__constant__ double lambda[3];

__global__
void symbol()
{
  printf("symbol : dx = %f %f %f\n", dx[0], dx[1], dx[2]);
  printf("symbol : n  = %d %d %d\n", n[0], n[1], n[2]);
}

void setDims(const int *h_n, 
             const double *h_xmin, 
             const double *h_dx, 
             const double *h_lambda)
{
    Timer & T = GetTimer(T_CopyId); T.start();
    cudaMemcpyToSymbol(n, h_n, 3 * sizeof(int));
    cudaMemcpyToSymbol(xmin, h_xmin, 3 * sizeof(double));
    cudaMemcpyToSymbol(dx, h_dx, 3 * sizeof(double));
    cudaMemcpyToSymbol(lambda, h_lambda, 3 * sizeof(double));
    T.stop();
  
    //symbol<<<1,1>>>();  
}