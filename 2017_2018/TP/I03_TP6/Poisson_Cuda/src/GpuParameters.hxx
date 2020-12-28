#ifndef GPUPARAMETERS_HXX_
#define GPUPARAMETERS_HXX_

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "AbstractParameters.hxx"

#ifdef __CUDACC__
#include <cuda.h>

struct sGPU {
	CUdevice device;
	CUcontext context;
	dim3 dimBlock, dimGrid;
};
#else
struct sGPU;
#endif

class GpuParameters : public virtual AbstractParameters {
public:

  GpuParameters(int argc, char **argv);
  ~GpuParameters();

  sGPU * GpuInfo;
};

#endif
