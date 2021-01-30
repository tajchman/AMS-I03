#include "reduce.h"
#include <vector>
#include "OpenCL.hxx"
#include "timer.hxx"

double reduce(int n, cl_mem d_w, OpenCL & CL)
{
  size_t local_size = 256; 
  size_t nGroups = (n + local_size)/local_size;
  size_t global_size = nGroups*local_size;

  cl_mem reductionBuffer = CL.allocate(nGroups);
  double sum;

  cl_kernel reductionKernel = CL.new_kernel("ReduceOpenCL", "reduce.cl");
  
  Timer & Tv = GetTimer(T_VerifId); Tv.start();
  Tv.start();

  cl_int errcode;
  
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
  sum = 0.0;
  for (int i=0; i<nGroups; i++)
      sum += sumReduction[i];
  Tv.stop();

  return sum;
}