#include "GpuScheme.hxx"
#include <math.h>
#include <iostream>
#include "CpuValues.hxx"
#include "cuda.h"

#define BLOCK_SIZE 512

GpuScheme::GpuScheme(const GpuParameters *P) : AbstractScheme(P),
 g_duv(P), g_duv2(P), m_w(P)
{

  codeName = "Poisson_GPU";
  deviceName = "GPU";

  m_u = new GpuValues(P);
  m_v = new GpuValues(P);
}

void GpuScheme::initialize()
{
  m_w.init();
  m_u->init();
  m_v->init();
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

  if (i>0 && i<nx-1 && j>0 && j<ny-1 && k>0 && k<nz-1) {
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

__global__  void
gpu_reduction(double * input, double * output, int len)
{
    __shared__ double partialSum[2*BLOCK_SIZE];
    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

    partialSum[t] =
    		((start + t) < len) ? input[start + t] : 0.0;

    partialSum[blockDim.x + t] =
    		((start + blockDim.x + t) < len) ? input[start + blockDim.x + t] : 0.0;

    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
    {
      __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t + stride];
    }
    __syncthreads();

    if (t == 0 && (globalThreadId*2) < len)
        output[blockIdx.x] = partialSum[t];
}

bool GpuScheme::iteration()
{
  const GpuParameters * p = dynamic_cast<const GpuParameters *>(m_P);
  const sGPU * g = p->GpuInfo;
  size_t n = m_n[0] * m_n[1] * m_n[2];

  gpu_iteration<<<g->dimBlock, g->dimGrid>>>
    (m_u->data(), m_v->data(), m_lambda, m_n[0], m_n[1], m_n[2]);
  cudaDeviceSynchronize();
  
  gpu_difference<<<g->dimBlock, g->dimGrid>>>
    (m_u->data(), m_v->data(), g_duv.data(), n);
  cudaDeviceSynchronize();

  int numOutputElements = n / (BLOCK_SIZE<<1);
  if (n % (BLOCK_SIZE<<1))
     numOutputElements++;

  double * dev_out = g_duv2.data();
  double * dev_in = g_duv.data();
  double * host_out = (double*) malloc(numOutputElements * sizeof(double));

  dim3 DimGrid( numOutputElements, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  gpu_reduction<<<DimGrid, DimBlock>>>(dev_in, dev_out, n);
  cudaMemcpy(host_out, dev_out, numOutputElements * sizeof(double), cudaMemcpyDeviceToHost);

  m_duv_max = 0.0;
  for (int i = 0; i < numOutputElements; i++)
   {
	  m_duv_max += host_out[i];
   }
  return true;
}

const AbstractValues & GpuScheme::getOutput()
{
  size_t n = m_n[0] * m_n[1] * m_n[2] * sizeof(double);
  cudaMemcpy(m_w.data(), m_u->data(), n, cudaMemcpyDeviceToHost);
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
    cudaMemcpy(m_u->data(), u1->data(), n, cudaMemcpyHostToDevice);
    return;
  }
  const GpuValues * u2 = dynamic_cast<const GpuValues *>(&u);
  std::cerr << "u2 = " << u2 << std::endl;    
  if (u2) {
    std::cerr << "setInput 1" << std::endl;
    cudaMemcpy(m_u->data(), u2->data(), n, cudaMemcpyDeviceToDevice);
    std::cerr << "setInput 2" << std::endl;
    return;
  }

}

