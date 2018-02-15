#include "GpuParameters.hxx"
#include "cuda.h"

GpuParameters::GpuParameters(int argc, char ** argv) : AbstractParameters(argc, argv)
{
		int deviceCount, devID;

		devID = 0;
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
		cuDeviceGet(&(GpuInfo->device), devID);
		cuCtxCreate(&(GpuInfo->context), 0, GpuInfo->device);

	    char deviceName[256];
        int major, minor;
        cuDeviceComputeCapability(&major, &minor, devID);
        cuDeviceGetName(deviceName, 256, devID);
        std::cerr << "Using Device " << devID
        		  << ": \"" << deviceName << "\" with Compute capability "
        		  << major << "." << minor << "\n";

        struct cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, devID);
        std::cerr<<"using "<<properties.multiProcessorCount<<" multiprocessors"<<std::endl;
        std::cerr<<"max threads per processor: "<<properties.maxThreadsPerMultiProcessor<<std::endl;

        int bx = properties.maxThreadsDim[0];
        int by = properties.maxThreadsDim[1];
        int bz = properties.maxThreadsDim[2];
        while (bx*by*bz > 256) {
        	if (bx >= bz && bx >= by)
        		bx /= 2;
        	else if (by >= bz && by >= bx)
                by /= 2;
        	else if (bz >= bx && bz >= by)
                bz /= 2;
        }
		GpuInfo->dimBlock = dim3(bx, by, bz);
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


