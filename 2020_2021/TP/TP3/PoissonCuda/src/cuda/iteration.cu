#include "dim.cuh"
#include "cuda_check.cuh"
#include "user.cuh"

#include "timer_id.hxx"
#include "iteration.hxx"

__global__
void iterKernel(double *v, double *u, double dt)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int p;
  double du, x, y, z;

  if (i>0 && i<d_n[0]-1 && j>0 && j<d_n[1]-1 && k>0 && k<d_n[2]-1) {
     p = i + d_n[0] * (j + k*d_n[1]);
     du = 0.0;
     du = (-2*u[p] + u[p+1]             + u[p-1])*d_lambda[0]
        + (-2*u[p] + u[p+d_n[0]]        + u[p-d_n[0]])*d_lambda[1]
        + (-2*u[p] + u[p+d_n[0]*d_n[1]] + u[p-d_n[0]*d_n[1]])*d_lambda[2];
     x = d_xmin[0] + i*d_dx[0];
     y = d_xmin[1] + j*d_dx[1];
     z = d_xmin[2] + k*d_dx[2];

     du += force(x,y,z);
    
     v[p] = u[p] + dt * du;
  }
}

void iterationWrapper(
    Values & v, Values & u, double dt, int n[3],
    int imin, int imax, 
    int jmin, int jmax,
    int kmin, int kmax)
{
  Timer & T = GetTimer(T_IterationId); T.start();

  dim3 dimBlock(8,8,8);
  dim3 dimGrid(int(ceil(n[0]/double(dimBlock.x))), 
               int(ceil(n[1]/double(dimBlock.y))),
               int(ceil(n[2]/double(dimBlock.z))));

  iterKernel<<<dimGrid, dimBlock>>>(v.dataGPU(), u.dataGPU(), dt);

  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();

  T.stop();
}
