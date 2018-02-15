#include "GpuScheme.hxx"
#include <math.h>
#include <iostream>
#include "CpuValues.hxx"
#include "cuda.h"
#include "ErrCheck.h"

#define BLOCK_SIZE 512

GpuScheme::GpuScheme(const GpuParameters *P) : AbstractScheme(P),
   host_out(NULL), dev_out(NULL), numOutputElements(0), m_u2(P), m_v2(P)
{

  codeName = "Poisson_GPU";
  deviceName = "GPU";

  m_u = new GpuValues(P);
  m_v = new GpuValues(P);

}

void GpuScheme::initialize()
{
  m_u->init();
  m_v->init();
  m_u2.init();
  m_v2.init();

//  numOutputElements = m_u->n_1D() / (BLOCK_SIZE<<1);
//  if (m_u->n_1D() % (BLOCK_SIZE<<1))
//     numOutputElements++;
//
//  host_out = (double*) malloc(numOutputElements * sizeof(double));
//  CHECK_CUDA(cudaMalloc(&dev_out, numOutputElements*sizeof(double)));

}

GpuScheme::~GpuScheme()
{
//  const GpuParameters * p = dynamic_cast<const GpuParameters *>(m_P);
//  const sGPU * g = p->GpuInfo;
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
gpu_zero(double *u, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x ;

  if (i<n)
    u[i] = 0.0;
}

__global__  void
gpu_norm(const double * input1, const double * input2, double * output, int len)
{
    __shared__ double partialSum[2*BLOCK_SIZE];
    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

    partialSum[t] =	((start + t) < len)
    		? (input1[start + t] - input2[start + t])
    		: 0.0;

    partialSum[blockDim.x + t] = ((start + blockDim.x + t) < len)
    		? (input1[start + blockDim.x + t] - input2[start + blockDim.x + t])
    		: 0.0;

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

  gpu_iteration<<<g->dimGrid, g->dimBlock>>>
    (m_u->data(), m_v->data(), m_lambda, m_n[0], m_n[1], m_n[2]);
  
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(m_v2.data(), m_v->data(), n * sizeof(double), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaDeviceSynchronize());

  m_duv_max = 0.0;
  double * a = m_u2.data();
  double * b = m_v2.data();
  for (int i = 0; i < n; i++)
   {
	  m_duv_max += fabs(a[i] -  b[i]);
   }
  m_u2.swap(m_v2);

  return true;
}

const AbstractValues & GpuScheme::getOutput()
{
  //size_t n = m_n[0] * m_n[1] * m_n[2] * sizeof(double);
  //CHECK_CUDA(cudaMemcpy(m_u2.data(), m_u->data(), n, cudaMemcpyDeviceToHost));
  return m_u2;
}

void GpuScheme::setInput(const AbstractValues & u)
{
  size_t n = m_n[0] * m_n[1] * m_n[2] * sizeof(double);

  const CpuValues * u1 = dynamic_cast<const CpuValues *>(&u);
  if (u1) {
	  memcpy(m_u2.data(), u1->data(), n);
	  CHECK_CUDA(cudaMemcpy(m_u->data(), u1->data(), n, cudaMemcpyHostToDevice));
    return;
  }
}

