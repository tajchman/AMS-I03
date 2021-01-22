#include <iostream>
#include "calcul_OpenCL.hxx"
#include "timer.hxx"


Calcul_OpenCL::Calcul_OpenCL(int m) : n(m)
{
  Timer & T = GetTimer(T_AllocId); T.start();
  
  d_u = CL.allocate(n);
  d_v = CL.allocate(n);
  d_w = CL.allocate(n);

  T.stop();
}

Calcul_OpenCL::~Calcul_OpenCL()
{
  Timer & T = GetTimer(T_FreeId); T.start();

  CL.deallocate(d_u);
  CL.deallocate(d_v);
  CL.deallocate(d_w);

  T.stop();
}

void Calcul_OpenCL::init()
{
  Timer & Tc = GetTimer(T_CompileId); Tc.start();

  cl_kernel initKernel = CL.new_kernel("InitOpenCL", "init.cl");
  
  Tc.stop();

  Timer & T = GetTimer(T_InitId); T.start();
  int errcode;
  
  errcode = clSetKernelArg(initKernel, 0, sizeof(cl_mem), &d_u);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(initKernel, 1, sizeof(cl_mem), &d_v);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(initKernel, 2, sizeof(int), &n);
  CheckOpenCL("clSetKernelArg");
  
  size_t local_size = 256; 
  size_t global_size = ((n + local_size)/local_size)*local_size;
  
  errcode = clEnqueueNDRangeKernel(CL.command_queue, initKernel, 
                          1, NULL, &global_size, &local_size,
			                    0, NULL, NULL);
  CheckOpenCL("clEnqueueNDRangeKernel");

  T.stop();
  CL.free_kernel(initKernel);
}


void Calcul_OpenCL::addition()
{
  Timer & Tc = GetTimer(T_CompileId); Tc.start();

  cl_kernel addKernel = CL.new_kernel("AddOpenCL", "add.cl");
  
  Tc.stop();

  Timer & T = GetTimer(T_AddId); T.start();

  int errcode;
  
  errcode = clSetKernelArg(addKernel, 0, sizeof(cl_mem), &d_w);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(addKernel, 1, sizeof(cl_mem), &d_u);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(addKernel, 2, sizeof(cl_mem), &d_v);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(addKernel, 3, sizeof(int), &n);
  CheckOpenCL("clSetKernelArg");
  
  size_t local_size = 256; 
  size_t global_size = ((n + local_size)/local_size)*local_size;
  
  errcode = clEnqueueNDRangeKernel(CL.command_queue, addKernel,
                          1, NULL, &global_size, &local_size,
			                    0, NULL, NULL);
  CheckOpenCL("clEnqueueNDRangeKernel");

  T.stop();

  CL.free_kernel(addKernel);

}

double Calcul_OpenCL::verification()
{
  Timer & T1 = GetTimer(T_CopyId);
  T1.start();
  
  int bytes = n*sizeof(double);
  std::vector<double> w(n);
  clEnqueueReadBuffer(CL.command_queue,
            d_w, CL_TRUE,
			       0, bytes, w.data(),
			       0, NULL, NULL);

  T1.stop();

  Timer & T = GetTimer(T_VerifId);
  T.start();

  double s = 0;
  for (int i=0; i<n; i++)
    s = s + w[i];
  
  s = s/n - 1.0;
  
  T.stop();

  return s;
}


