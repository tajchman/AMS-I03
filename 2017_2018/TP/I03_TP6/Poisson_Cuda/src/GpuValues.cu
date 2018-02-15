#include "GpuValues.hxx"
#include "f.hxx"
#include "ErrCheck.h"

GpuValues::GpuValues(const GpuParameters * p) : AbstractValues(p)
{
}

__global__ void
gpu_init(double *u, size_t nx, size_t ny, size_t nz) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x ;
	const int j = blockIdx.y * blockDim.y + threadIdx.y ;
	const int k = blockIdx.z * blockDim.z + threadIdx.z ;

	const int i_j_k  = i + j*nx + k*nx*ny;
	if (i<nx && j<ny && k<nz)
		u[i_j_k] = 0.;
}

__global__ void
gpu_init_f(double *u, size_t nx, size_t ny, size_t nz,
		double xmin, double ymin, double zmin,
		double dx,   double dy,   double dz) {

	const int i = blockIdx.x * blockDim.x + threadIdx.x ;
	const int j = blockIdx.y * blockDim.y + threadIdx.y ;
	const int k = blockIdx.z * blockDim.z + threadIdx.z ;

	const int i_j_k  = i + j*nx + k*nx*ny;
	if (i<nx && j<ny && k<nz) {
	   u[i_j_k] = f_GPU(xmin + i*dx, ymin + j*dx, zmin + k*dz);
 	   }
}

void GpuValues::init_f()
{
  allocate(nn);

  const GpuParameters * p = dynamic_cast<const GpuParameters *>(m_p);
  const sGPU * g = p->GpuInfo;

  gpu_init_f<<<g->dimBlock, g->dimGrid>>>(m_u,
	p->n(0),    p->n(1),    p->n(2),
	p->xmin(0), p->xmin(1), p->xmin(2),
	p->dx(0),   p->dx(1),   p->dx(2));
  CHECK_CUDA(cudaDeviceSynchronize());
}

void GpuValues::init()
{
  allocate(nn);
}


void GpuValues::allocate(size_t n) {
	deallocate();
	CHECK_CUDA(cudaMalloc(&m_u, n*sizeof(double)));
}

void GpuValues::deallocate() {
	if (m_u != NULL) {
		CHECK_CUDA(cudaFree(m_u));
		m_u = NULL;
	}
}
