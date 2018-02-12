#include "GPU.hxx"
#include "GpuScheme.hxx"
#include <math.h>
#include <cuda.h>
#include <iostream>

struct sGPU {

  CUdevice device;
  CUcontext context;
  dim3 dimBlock, dimGrid;
};

GpuScheme::GpuScheme(const Parameters *P) : Scheme(P), m_duv(P), m_duv2(P)  {
  codeName = "Poisson_Cuda";
  deviceName = "GPU";
  m_GPU = new sGPU;

  int deviceCount;
  CHECK_CUDA_RESULT(cuInit(0));
  CHECK_CUDA_RESULT(cuDeviceGetCount(&deviceCount));
  if (deviceCount == 0) {
    std::cerr << "GPU : no CUDA device found" << std::endl;
    exit(1);
  }
  else {
    std::cerr << "GPU : " << deviceCount << " CUDA device";
    if (deviceCount > 1) std::cerr << "s";
    std::cerr << " found\n" << std::endl;
  }
  CHECK_CUDA_RESULT(cuDeviceGet(&(m_GPU->device), 0));
  CHECK_CUDA_RESULT(cuCtxCreate(&(m_GPU->context), 0, m_GPU->device));

#define BLOCK_SIZE_X 4
#define BLOCK_SIZE_Y 4
#define BLOCK_SIZE_Z 4

  m_GPU->dimBlock = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  m_GPU->dimGrid  = dim3( ceil(float(m_n[0])/float(m_GPU->dimBlock.x)),
			  ceil(float(m_n[1])/float(m_GPU->dimBlock.y)),
			  ceil(float(m_n[2])/float(m_GPU->dimBlock.z)));

  std::cerr << "Block "
            << m_GPU->dimBlock.x << " x "
            << m_GPU->dimBlock.y << " x "
            << m_GPU->dimBlock.z << std::endl;
  std::cerr << "Grid  "
            << m_GPU->dimGrid.x << " x "
            << m_GPU->dimGrid.y << " x "
            << m_GPU->dimGrid.z << std::endl << std::endl;
}

void GpuScheme::initialize()
{
  Scheme::initialize();
  m_duv.init();
}

GpuScheme::~GpuScheme()
{
  CHECK_CUDA_RESULT(cuCtxDestroy(m_GPU->context));
}

__global__ void
gpu_iteration(const double *u, double *v, double lambda, int nx, int ny, int nz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x ;
  const int j = blockIdx.y * blockDim.y + threadIdx.y ;
  const int k = blockIdx.z * blockDim.z + threadIdx.z ;

  int i_j_k  = i + j*nx + k*nx*ny;

  int im_j_k = nx > 2 ? i+1 + j*nx + k*nx*ny : i_j_k;
  int ip_j_k = nx > 2 ? i-1 + j*nx + k*nx*ny : i_j_k;

  int i_jm_k = ny > 2 ? i + (j-1)*nx + k*nx*ny : i_j_k;
  int i_jp_k = ny > 2 ? i + (j+1)*nx + k*nx*ny : i_j_k;

  int i_j_km = nz > 2 ? i + j*nx + (k-1)*nx*ny : i_j_k;
  int i_j_kp = nz > 2 ? i + j*nx + (k+1)*nx*ny : i_j_k;

//  printf(" %d %d %d ---- %d %d %d === %d \n", i>0, j>0, k>0, i<nx-1, j<ny-1, k<nz-1, i>0 && i<nx-1 && j>0 && j<ny-1 && k>0 && k<nz-1);
  if (i>0 && i<nx-1 && j>0 && j<ny-1 && k>0 && k<nz-1) {
	    printf("centre u(%d,%d,%d) = %d\n", i,j,k, u[i_j_k]);
    v[i_j_k] = u[i_j_k] - lambda *
      (6 * u[i_j_k] - u[ip_j_k] - u[im_j_k]
       - u[i_jp_k] - u[i_jm_k]
       - u[i_j_kp] - u[i_j_km]);
  }
}

__global__ void
gpu_difference(const double *u, const double *v, double * duv,
	       int nx, int ny, int nz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x ;
  const int j = blockIdx.y * blockDim.y + threadIdx.y ;
  const int k = blockIdx.z * blockDim.z + threadIdx.z ;

  int i_j_k  = i + j*nx + k*nx*ny;

  if (i>0 && i<nx && j>0 && j<ny && k>0 && k<nz)
    duv[i_j_k] = fabs(v[i_j_k] - u[i_j_k]);
}

__global__ void gpu_reduction(double *g_idata, double *g_odata, int n)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[tid] = i<n ? g_idata[i] : 0.0;
  __syncthreads();
  // do reduction in shared mem
  for(unsigned int s=1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


bool GpuScheme::iteration()
{

//  std::cerr << "u:" << std::endl;
//  m_u.print(std::cerr);
  
  cudaDeviceSynchronize();
  gpu_iteration<<<m_GPU->dimBlock, m_GPU->dimGrid>>>
    (m_u.data(), m_v.data(), m_lambda, m_n[0]-1, m_n[1]-1, m_n[2]-1);
  cudaDeviceSynchronize();
  std::exit(-1);
  
  gpu_difference<<<m_GPU->dimBlock, m_GPU->dimGrid>>>
    (m_u.data(), m_v.data(), m_duv.data(), m_n[0], m_n[1], m_n[2]);
  cudaDeviceSynchronize();
  gpu_reduction<<<m_GPU->dimBlock, m_GPU->dimGrid>>>
    (m_duv.data(), m_duv2.data(), m_n[0]*m_n[1]*m_n[2]);

  m_duv_max = m_duv2.data()[0];
  std::cerr << m_duv_max << std::endl;
  return true;
}

