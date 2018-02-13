#include "GpuValues.hxx"
#include "f.hxx"

GpuValues::GpuValues(const GpuParameters * p) : AbstractValues(p), g_u(NULL)
{
}

__global__ void
gpu_init0(double *u, size_t nx, size_t ny, size_t nz) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x ;
	const int j = blockIdx.y * blockDim.y + threadIdx.y ;
	const int k = blockIdx.z * blockDim.z + threadIdx.z ;

	int i_j_k  = i + j*nx + k*nx*ny;
	if (i>0 && i<nx-1 && j>0 && j<ny-1 && k>0 && k<nz-1)
		u[i_j_k] = 0.;
}

__global__ void
gpu_init(double *u, size_t nx, size_t ny, size_t nz,
		double xmin, double ymin, double zmin,
		double dx,   double dy,   double dz) {

	const int i = blockIdx.x * blockDim.x + threadIdx.x ;
	const int j = blockIdx.y * blockDim.y + threadIdx.y ;
	const int k = blockIdx.z * blockDim.z + threadIdx.z ;

	int i_j_k  = i + j*nx + k*nx*ny;
	if (i>0 && i<nx-1 && j>0 && j<ny-1 && k>0 && k<nz-1) {
		u[i_j_k] = f_GPU(xmin + i*dx, ymin + j*dx, zmin + k*dz);
		printf("u %d %d %d = %g\n", i,j,k, u[i_j_k]);
	}
}

void GpuValues::init_f()
{
	allocate(nn);

	const GpuParameters * p = dynamic_cast<const GpuParameters *>(m_p);
	const sGPU * g = p->GpuInfo;

	gpu_init<<<g->dimBlock, g->dimGrid>>>
			(m_u,
					p->n(0),    p->n(1),    p->n(2),
					p->xmin(0), p->xmin(1), p->xmin(2),
					p->dx(0),   p->dx(1),   p->dx(2));
}

void GpuValues::init()
{
	allocate(nn);

	const GpuParameters * p = dynamic_cast<const GpuParameters *>(m_p);
	const sGPU * g = p->GpuInfo;

	gpu_init0<<<g->dimBlock, g->dimGrid>>>
			(m_u,
					p->n(0),    p->n(1),    p->n(2));
}


void GpuValues::allocate(size_t n) {
	deallocate();
	m_u = new double[n];
	cudaMalloc(&g_u, n*sizeof(double));
}

void GpuValues::deallocate() {
	if (m_u != NULL) {
		delete [] m_u;
		m_u = NULL;
	}
	if (g_u != NULL) {
		cudaFree(g_u);
		g_u = NULL;
	}
}
