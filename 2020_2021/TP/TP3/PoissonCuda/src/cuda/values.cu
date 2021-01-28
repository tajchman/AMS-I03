#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <cuda.h>

#include "cuda_check.cuh"
#include "dim.cuh"
#include "user.cuh"

#include "values.hxx"
#include "os.hxx"
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

__global__
void zeroValue(double *u, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i<n) {
    u[i] = 0.0;
  }
}

void zeroWrapper(double *d, int n)
{
  int dimBlock = 256;
  int dimGrid = (n + dimBlock - 1)/dimBlock;

  zeroValue<<<dimGrid, dimBlock>>>(d, n);
  CUDA_CHECK_KERNEL();
}


__global__
void initValue(double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int p;

  if (i>0 && i<n[0]-1 && j>0 && j<n[1]-1 && k>0 && k<n[2]-1)
  {
    p = i + n[0] * (j + k*n[1]);
    u[p] = cond_ini(xmin[0] + i*dx[0],
                    xmin[1] + j*dx[1], 
                    xmin[2] + k*dx[2]);
  }
}

void initWrapper(double *d, int n[3])
{
  Timer & T = GetTimer(T_InitId); T.start();

  dim3 dimBlock(8,8,8);
  dim3 dimGrid(int(ceil(n[0]/double(dimBlock.x))),
               int(ceil(n[1]/double(dimBlock.y))),
               int(ceil(n[2]/double(dimBlock.z))));

  initValue<<<dimGrid, dimBlock>>>(d);
  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();
  
  T.stop();
}


__global__
void boundZValue(int k, double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int p;

  if (i<n[0] && j<n[1]) {
    p = i + j*n[0] + k*n[0]*n[1];
    u[p] = cond_lim(xmin[0] + i*dx[0], 
                    xmin[1] + j*dx[1], 
                    xmin[2] + k*dx[2]);
  }
}


__global__
void boundYValue(int j, double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int p;

  if (i<n[0] && k<n[2]) {
    p = i + j*n[0] + k*n[0]*n[1];
    u[p] = cond_lim(xmin[0] + i*dx[0], 
                    xmin[1] + j*dx[1], 
                    xmin[2] + k*dx[2]);
  }
}

__global__
void boundXValue(int i, double *u)
{
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int p;

  if (j<n[1] && k<n[2]) {
    p = i + j*n[0] + k*n[0]*n[1];
    u[p] = cond_lim(xmin[0] + i*dx[0], 
                    xmin[1] + j*dx[1], 
                    xmin[2] + k*dx[2]);
  }
}

void boundariesWrapper(double *d, int n[3], int imin[3], int imax[3])
{
  Timer & T = GetTimer(T_InitId); T.start();

  dim3 dimBlock2(16,16,1);
  dim3 dimGrid2(int(ceil(n[0]/double(dimBlock2.x))),
                int(ceil(n[1]/double(dimBlock2.y))), 
                1);
  boundZValue<<<dimGrid2, dimBlock2>>>(imin[2]-1, d);
  boundZValue<<<dimGrid2, dimBlock2>>>(imax[2]+1, d);

  dim3 dimBlock1(16,1,16);
  dim3 dimGrid1(int(ceil(n[0]/double(dimBlock1.x))), 
                1,
                int(ceil(n[2]/double(dimBlock1.z))));

  boundYValue<<<dimGrid1, dimBlock1>>>(imin[1]-1, d);
  boundYValue<<<dimGrid1, dimBlock1>>>(imax[1]+1, d);

  dim3 dimBlock0(1,16,16);
  dim3 dimGrid0(1, 
                int(ceil(n[1]/double(dimBlock0.y))),
                int(ceil(n[2]/double(dimBlock0.z))));

  boundXValue<<<dimGrid0, dimBlock0>>>(imin[0]-1, d);
  boundXValue<<<dimGrid0, dimBlock0>>>(imax[0]+1, d);

  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();

  T.stop();
}
