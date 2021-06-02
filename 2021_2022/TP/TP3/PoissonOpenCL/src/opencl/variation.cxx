
#include "variation.hxx"

#include <stdio.h>
#include <iostream>
#include <vector>
#include "timer_id.hxx"
#include "OpenCL.hxx"
#include "values.hxx"

void allocVariationData(double *& diff, int n,
                        double *& diffPartial, int nPartial,
                        OpenCL & CL)
{
if (diff == NULL) 
  diff = CL.allocate(n);
if (diffPartial == NULL)
  diffPartial = CL.allocate(nPartial);
}

void freeVariationData(double *& diff, 
                       double *& diffPartial,
                        OpenCL & CL)
{
  CL.deallocate(diff);
  CL.deallocate(diffPartial);
}

const char * sDifferenceKernel = "\
__kernel void\
difference (__global const double *u, \
            __global const double *v, \
            __global double * duv, int n)\
{\
  const int i = get_global_id (0);\
  if (i<n) {\
    duv[i] = fabs(u[i] - v[i]);\
  }\
}\
";

const char * sReduceKernel = "\
__kernel void\
reduce(__global const double *input,\
             __global double *partialSums,\
             __local double *localSums, int n)\
{\
   uint local_id = get_local_id(0);\
   uint group_size = get_local_size(0);\
   int global_id = get_global_id(0);\
\
   // Copy from global memory to local memory\
   if  (global_id < n)\
     localSums[local_id] = input[get_global_id(0)];\
   else\
     localSums[local_id] = 0.0;\
\
   for (uint stride = group_size/2; stride>0; stride/=2) {\
       barrier(CLK_LOCAL_MEM_FENCE);\
\
      if (local_id < stride)\
        localSums[local_id] += localSums[local_id + stride];\
   }\
\
   if (local_id == 0)\
     partialSums[get_group_id(0)] = localSums[0];\
}\
";

double variationWrapper(const Values &u, 
                        const Values &v,
                        double * &diff,
                        double * &diffPartial, int n,
                        OpenCL & CL)
{
  size_t local_size = 256; 
  size_t nGroups = (n + local_size)/local_size;
  size_t global_size = nGroups*local_size;
  
  int nbytesWork = nGroups * sizeof(double);
  int nbytesGroup = local_size * sizeof(double);
  int nbytes = n * sizeof(double);

  allocVariationData(diff,        nbytes, 
                     diffPartial, nbytesWork, CL);

  Timer & Tv = GetTimer(T_VariationId); Tv.start();

  cl_kernel differenceKernel 
      = CL.new_kernel_source("difference", sDifferenceKernel);
  cl_kernel reductionKernel
      = CL.new_kernel_source("reduce", sReduceKernel);
  
  Timer & Tv = GetTimer(T_VerifId); Tv.start();
  Tv.start();

  cl_int errcode;
  
  errcode = clSetKernelArg(differenceKernel, 0, sizeof(cl_mem), &diff);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(differenceKernel, 1, sizeof(cl_mem), &diffPartial);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(differenceKernel, 2, size(int), &n);
  CheckOpenCL("clSetKernelArg");

  errcode = clEnqueueNDRangeKernel(CL.command_queue, differenceKernel,
                                   1, NULL, &global_size, &local_size,
			                             0, NULL, NULL);
  CheckOpenCL("clEnqueueNDRangeKernel");

  errcode = clSetKernelArg(reductionKernel, 0, sizeof(cl_mem), &d_w);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(reductionKernel, 1, sizeof(cl_mem), &reductionBuffer);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(reductionKernel, 2, local_size * sizeof(double), NULL);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(reductionKernel, 3, sizeof(int), &n);
  CheckOpenCL("clSetKernelArg");
  
  errcode = clEnqueueNDRangeKernel(CL.command_queue, reductionKernel,
                                   1, NULL, &global_size, &local_size,
			                             0, NULL, NULL);
  CheckOpenCL("clEnqueueNDRangeKernel");

  Tv.stop();

  std::vector<double> sumReduction(nGroups);
  CL.memcpyDeviceToHost(reductionBuffer, sumReduction.data(), nGroups);
  CL.free_kernel(reductionKernel);
  CL.deallocate(reductionBuffer);

  Tv.start();
  double sum = 0.0;
  for (int i=0; i<nGroups; i++)
      sum += sumReduction[i];
  Tv.stop();

  return sum;
}

