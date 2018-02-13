#include "GpuScheme.hxx"
#include <math.h>
#include <iostream>
#include "CpuValues.hxx"
#include "cuda.h"

GpuScheme::GpuScheme(const GpuParameters *P) : AbstractScheme(P), m_duv(P), m_duv2(P)  {
	m_u = new GpuValues(P);
	m_v = new GpuValues(P);
	m_w = new CpuValues(P);

  codeName = "Poisson_GPU";
  deviceName = "GPU";

}

void GpuScheme::initialize()
{
  AbstractScheme::initialize();
  m_duv.init();
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

//  printf(" %d %d %d ---- %d %d %d === %d \n", i>0, j>0, k>0, i<nx-1, j<ny-1, k<nz-1, i>0 && i<nx-1 && j>0 && j<ny-1 && k>0 && k<nz-1);
  if (i>0 && i<nx-1 && j>0 && j<ny-1 && k>0 && k<nz-1) {
	    printf("centre u(%d,%d,%d) = %g\n", i,j,k, u[i_j_k]);
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

  const GpuParameters * p = dynamic_cast<const GpuParameters *>(m_P);
  const sGPU * g = p->GpuInfo;

//  std::cerr << "u:" << std::endl;
//  m_u.print(std::cerr);
  
  cudaDeviceSynchronize();
  gpu_iteration<<<g->dimBlock, g->dimGrid>>>
    (m_u->data(), m_v->data(), m_lambda, m_n[0]-1, m_n[1]-1, m_n[2]-1);
  cudaDeviceSynchronize();
  std::exit(-1);
  
  gpu_difference<<<g->dimBlock, g->dimGrid>>>
    (m_u->data(), m_v->data(), m_duv.data(), m_n[0], m_n[1], m_n[2]);
  cudaDeviceSynchronize();
  gpu_reduction<<<g->dimBlock, g->dimGrid>>>
    (m_duv.data(), m_duv2.data(), m_n[0]*m_n[1]*m_n[2]);

  m_duv_max = m_duv2.data()[0];
  std::cerr << m_duv_max << std::endl;
  return true;
}

const AbstractValues & GpuScheme::getOutput()
{
  m_w->init();

  size_t n = m_n[0] * m_n[1] * m_n[2] * sizeof(double);
  cudaMemcpy(m_w->data(), m_u->data(), n, cudaMemcpyDeviceToHost);
  return *m_w;
}

void GpuScheme::setInput(const AbstractValues & u)
{
	  size_t n = m_n[0] * m_n[1] * m_n[2] * sizeof(double);
	  cudaMemcpy(m_u->data(), u.data(), n, cudaMemcpyDeviceToDevice);

}

