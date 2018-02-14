#include "GpuScheme.hxx"
#include <math.h>
#include <iostream>
#include "CpuValues.hxx"
#include "cuda.h"

GpuScheme::GpuScheme(const GpuParameters *P) : AbstractScheme(P),
 g_u(P), g_v(P), g_duv(P), g_duv2(P), m_w(P) 
{

  codeName = "Poisson_GPU";
  deviceName = "GPU";

}

void GpuScheme::initialize()
{
  m_w.init();
  g_u.init();
  g_v.init();
  g_duv.init();
  g_duv2.init();
}

GpuScheme::~GpuScheme()
{
  const GpuParameters * p = dynamic_cast<const GpuParameters *>(m_P);
  const sGPU * g = p->GpuInfo;
  cuCtxDestroy(g->context);
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

  if (i<nx-1 && j<ny-1 && k<nz-1) {
	    printf("centre u(%d,%d,%d) = %g\n", i,j,k, u[i_j_k]);
    v[i_j_k] = u[i_j_k] - lambda *
      (6 * u[i_j_k] - u[ip_j_k] - u[im_j_k]
       - u[i_jp_k] - u[i_jm_k]
       - u[i_j_kp] - u[i_j_km]);
  }
}

__global__ void
gpu_difference(const double *u, const double *v, double * duv,
	       int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x ;

  if (i<n)
    duv[i] = fabs(v[i] - u[i]);
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
  const GpuParameters * p = dynamic_cast<const GpuParameters *>(m_P);
  const sGPU * g = p->GpuInfo;
  size_t n = m_n[0] * m_n[1] * m_n[2];

//  std::cerr << "u:" << std::endl;
//  m_u.print(std::cerr);
  
  gpu_iteration<<<g->dimBlock, g->dimGrid>>>
    (g_u.data(), g_v.data(), m_lambda, m_n[0], m_n[1], m_n[2]);
  cudaDeviceSynchronize();
  std::exit(-1);
  
  gpu_difference<<<g->dimBlock, g->dimGrid>>>
    (g_u.data(), g_v.data(), g_duv.data(), n);
  cudaDeviceSynchronize();

  gpu_reduction<<<g->dimBlock, g->dimGrid>>>
    (g_duv.data(), g_duv2.data(), n);
  cudaDeviceSynchronize();

  cudaMemcpy(&m_duv_max, g_duv2.data(), sizeof(double), cudaMemcpyDeviceToHost);
  std::cerr << m_duv_max << std::endl;
  return true;
}

const AbstractValues & GpuScheme::getOutput()
{
  size_t n = m_n[0] * m_n[1] * m_n[2] * sizeof(double);
  cudaMemcpy(m_w.data(), g_u.data(), n, cudaMemcpyDeviceToHost);
  std::cerr << m_w(6,4,5) << std::endl;
  return m_w;
}

void GpuScheme::setInput(const AbstractValues & u)
{
  size_t n = m_n[0] * m_n[1] * m_n[2] * sizeof(double);
  std::cerr << "n = " << n << std::endl;    

  const CpuValues * u1 = dynamic_cast<const CpuValues *>(&u);
  std::cerr << "u1 = " << u1 << std::endl;    
  if (u1) {
    cudaMemcpy(g_u.data(), u1->data(), n, cudaMemcpyHostToDevice);
    return;
  }
  const GpuValues * u2 = dynamic_cast<const GpuValues *>(&u);
  std::cerr << "u2 = " << u2 << std::endl;    
  if (u2) {
    std::cerr << "setInput 1" << std::endl;
    cudaMemcpy(g_u.data(), u2->data(), n, cudaMemcpyDeviceToDevice);
    std::cerr << "setInput 2" << std::endl;
    return;
  }

}

