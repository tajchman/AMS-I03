#include "GpuParameters.hxx"
#include "cuda.h"

GpuParameters::GpuParameters(int argc, char ** argv) : AbstractParameters(argc, argv)
{
		int deviceCount, devID;
		cudaDeviceProp deviceProps;

		devID = 0;
		cudaGetDeviceCount(&deviceCount);
		if (deviceCount == 0) {
			std::cerr << "GPU : no CUDA device found" << std::endl;
			exit(1);
		}
		else {
			std::cerr << "GPU : " << deviceCount << " CUDA device";
			if (deviceCount > 1) std::cerr << "s";
			std::cerr << " found\n" << std::endl;
		}

		cudaSetDevice(devID);
		cudaGetDeviceProperties(&deviceProps, devID);
		std::cerr << "CUDA device [" << deviceProps.name << "] has "
				  << deviceProps.multiProcessorCount << " Multi-Processors\n";

		cudaSetDevice(devID);

		GpuInfo = new sGPU;
		GpuInfo->device = devID;

     std::cerr<<"using "<<deviceProps.multiProcessorCount<<" multiprocessors"<<std::endl;
        std::cerr<<"max threads per processor: "<<deviceProps.maxThreadsPerMultiProcessor<<std::endl;

        int bx = deviceProps.maxThreadsDim[0]; bx = 128;
        int by = deviceProps.maxThreadsDim[1]; by = 4;
        int bz = deviceProps.maxThreadsDim[2]; bz = 1;
        while (bx*by*bz > 512) {
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


