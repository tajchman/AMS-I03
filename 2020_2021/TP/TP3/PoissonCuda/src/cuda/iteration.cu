#include "timer_id.hxx"
#include "iteration.hxx"
#include "dim.cuh"
#include "cuda_check.cuh"

__device__
double f(double x, double y, double z)
{
  return (x < 0.3) ? 0.0 : sin(x-0.5) * cos(y-0.5) * exp(- z*z);
}

__global__
void iterKernel(double *v, double *u, double dt)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int p;
  double du, x, y, z;

  if (i>0 && i<n[0]-1 && j>0 && j<n[1]-1 && k>0 && k<n[2]-1) {
     p = i + n[0] * (j + k*n[1]);
     du = (- 2*u[p] + u[p + 1]         + u[p - 1])*lambda[0]
        + (- 2*u[p] + u[p + n[0]]      + u[p - n[0]])*lambda[1]
        + (- 2*u[p] + u[p + n[0]*n[1]] + u[p - n[0]*n[1]])*lambda[2];

     x = xmin[0] + i*dx[0];
     y = xmin[1] + j*dx[1];
     z = xmin[2] + k*dx[2];

     du += f(x,y,z);
    
     v[p] = u[p] + dt * du;
  }
}

void iterationWrapper(
    Values & v, Values & u, double dt, int n[3],
    int imin, int imax, 
    int jmin, int jmax,
    int kmin, int kmax)
{
  dim3 dimBlock(8,8,8);
  dim3 dimGrid(ceil(n[0]/double(dimBlock.x)), 
  ceil(n[1]/double(dimBlock.y)),
  ceil(n[2]/double(dimBlock.z)));

  Timer & T = GetTimer(T_IterationId); T.start();

  iterKernel<<<dimGrid, dimBlock>>>(v.dataGPU(), u.dataGPU(), dt);
  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();

  T.stop();
}
