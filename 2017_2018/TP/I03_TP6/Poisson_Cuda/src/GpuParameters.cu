#include "GpuParameters.hxx"
#include "cuda.h"

GpuParameters::GpuParameters(int argc, char ** argv) : AbstractParameters(argc, argv)
{
		int deviceCount;
		cuInit(0);
		cuDeviceGetCount(&deviceCount);
		if (deviceCount == 0) {
			std::cerr << "GPU : no CUDA device found" << std::endl;
			exit(1);
		}
		else {
			std::cerr << "GPU : " << deviceCount << " CUDA device";
			if (deviceCount > 1) std::cerr << "s";
			std::cerr << " found\n" << std::endl;
		}

		GpuInfo = new sGPU;
		cuDeviceGet(&(GpuInfo->device), 0);
		cuCtxCreate(&(GpuInfo->context), 0, GpuInfo->device);

#define BLOCK_SIZE_X 4
#define BLOCK_SIZE_Y 4
#define BLOCK_SIZE_Z 4

		GpuInfo->dimBlock = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
		GpuInfo->dimGrid  = dim3(
				ceil(float(m_n[0])/float(GpuInfo->dimBlock.x)),
				ceil(float(m_n[1])/float(GpuInfo->dimBlock.y)),
				ceil(float(m_n[2])/float(GpuInfo->dimBlock.z)));

		std::cerr << "Block "
				<< GpuInfo->dimBlock.x << " x "
				<< GpuInfo->dimBlock.y << " x "
				<< GpuInfo->dimBlock.z << std::endl;
		std::cerr << "Grid  "
				<< GpuInfo->dimGrid.x << " x "
				<< GpuInfo->dimGrid.y << " x "
				<< GpuInfo->dimGrid.z << std::endl << std::endl;
}

GpuParameters::~GpuParameters()
{
}


