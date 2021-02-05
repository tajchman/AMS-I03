#include <iostream>
#include "Calcul_MPI_Cuda.hxx"
#include "timer.hxx"
#include "reduction.h"
#include "cuda_check.cuh"

Calcul_MPI_Cuda::Calcul_MPI_Cuda(int m0, int m1, int rank)
{
  Timer & T = GetTimer(T_AllocId); T.start();
  
  int nDevices;
  CUDA_CHECK_OP(cudaGetDeviceCount(&nDevices));
  CUDA_CHECK_OP(cudaSetDevice(rank%nDevices));
  std::cerr << "Using device " << rank%nDevices << "/" << nDevices << std::endl;

  n0 = m0; n1 = m1;
  n_local = n1-n0;
  int bytes = n_local*sizeof(double);
  CUDA_CHECK_OP(cudaMalloc(&d_u, bytes));
  CUDA_CHECK_OP(cudaMalloc(&d_v, bytes));
  CUDA_CHECK_OP(cudaMalloc(&d_w, bytes));
  CUDA_CHECK_OP(cudaMalloc(&d_tmp, bytes));

  T.stop();
}

Calcul_MPI_Cuda::~Calcul_MPI_Cuda()
{
  Timer & T = GetTimer(T_FreeId); T.start();

  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_tmp);

  T.stop();
}

__global__ void vecInit(double *a, double *b, int n, int n0)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  double x;
  
  if (id >= n) return;

  x = double(n0 + id);
  a[id] = sin(x)*sin(x);
  b[id] = cos(x)*cos(x);
}

void Calcul_MPI_Cuda::init()
{
  Timer & T = GetTimer(T_InitId); T.start();

  blockSize = 512;
  gridSize = (unsigned int) ceil((double)n_local/blockSize);
  
  vecInit<<<gridSize, blockSize>>>(d_u, d_v, n_local, n0);
  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();

  T.stop();
}

__global__ void vecAdd(double *c, double *a, double *b, int n)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (id < n)
    c[id] = a[id] + b[id];
}

void Calcul_MPI_Cuda::addition()
{
  Timer & T = GetTimer(T_AddId); T.start();
  
  vecAdd<<<gridSize, blockSize>>>(d_w, d_u, d_v, n_local);
  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();

  T.stop();
}

double Calcul_MPI_Cuda::somme()
{
  Timer & T = GetTimer(T_SommeId);
  T.start();
  
  double s;
  s = reduce(n_local, d_w, d_tmp, blockSize, gridSize);  
  
  T.stop();

  return s;
}

