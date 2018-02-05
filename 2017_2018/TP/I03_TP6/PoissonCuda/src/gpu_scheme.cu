#include "GPU.hxx"
#include "gpu_scheme.hxx"
#include <math.h>
#include <cuda.h>
#include <iostream>

struct sGPU {

	CUdevice device;
	CUcontext context;
	dim3 dimBlock, dimGrid;
};

GPUScheme::GPUScheme()
{
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

    m_GPU->dimBlock = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    m_GPU->dimGrid  = dim3( ceil(float(m_n[0])/float(m_GPU->dimBlock.x)),
                        ceil(float(m_n[1])/float(m_GPU->dimBlock.y)),
                        ceil(float(m_n[2])/float(m_GPU->dimBlock.z)));
}

GPUScheme::~GPUScheme()
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

    if (i>0 && i<nx && j>0 && j<ny && k>0 && k<nz)
    	v[i_j_k] = u[i_j_k] - lambda *
    	    (6 * u[i_j_k] - u[ip_j_k] - u[im_j_k]
                          - u[i_jp_k] - u[i_jm_k]
                          - u[i_j_kp] - u[i_j_km]);
}

__global__ void
gpu_difference(const double *u, const double *v, int nx, int ny, int nz)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x ;
	const int j = blockIdx.y * blockDim.y + threadIdx.y ;
	const int k = blockIdx.z * blockDim.z + threadIdx.z ;

	int i_j_k  = i + j*nx + k*nx*ny;

    double du = 0.0;

	if (i>0 && i<nx && j>0 && j<ny && k>0 && k<nz)
       du = fabs(v[i_j_k] - u[i_j_k]);
}


bool GPUScheme::iteration()
{

   gpu_iteration<<<m_GPU->dimBlock, m_GPU->dimGrid>>>(m_u.data(), m_v.data(),
		                                          m_lambda, m_n[0], m_n[1], m_n[2]);
   gpu_difference<<<m_GPU->dimBlock, m_GPU->dimGrid>>>(m_u.data(), m_v.data(),
		                                          m_n[0], m_n[1], m_n[2]);
   return true;
}

