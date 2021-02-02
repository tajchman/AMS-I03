#include "OpenCL.hxx"

#include "timer_id.hxx"
#include "iteration.hxx"

const char * sIterKernel = "\
__kernel\
void iterKernel(__global double *v, __global double *u, double dt, \
                __global double *d_xmin, __global double *d_dx, \
                __global double *d_lambda, __global *d_n)\
{\
  const int i = get_global_id (0);\
  const int j = get_global_id (1);\
  const int k = get_global_id (2);\
  int p;\
  double du, x, y, z;\
\
  if (i>0 && i<d_n[0]-1 && j>0 && j<d_n[1]-1 && k>0 && k<d_n[2]-1) {\
     p = i + d_n[0] * (j + k*d_n[1]);\
     du = 0.0;\
     du = (-2*u[p] + u[p+1]             + u[p-1])*d_lambda[0]\
        + (-2*u[p] + u[p+d_n[0]]        + u[p-d_n[0]])*d_lambda[1]\
        + (-2*u[p] + u[p+d_n[0]*d_n[1]] + u[p-d_n[0]*d_n[1]])*d_lambda[2];\
     x = d_xmin[0] + i*d_dx[0];\
     y = d_xmin[1] + j*d_dx[1];\
     z = d_xmin[2] + k*d_dx[2];\
    \
     du += force(x,y,z);\
    \
     v[p] = u[p] + dt * du;\
  }\
}\
";


void iterationWrapper(
    Values & v, Values & u, double dt, int n[3],
    cl_kernel kern, OpenCL & CL)
{
  Timer & T = GetTimer(T_IterationId); T.start();

  int errcode;
  
  errcode = clSetKernelArg(kern, 0, sizeof(cl_mem), &v.dataGPU());
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 1, sizeof(cl_mem), &u.dataGPU());
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 2, sizeof(double), &dt);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 3, sizeof(cl_mem), &dt);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 4, sizeof(cl_mem), &dt);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 5, sizeof(cl_mem), &dt);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 6, sizeof(cl_mem), &dt);
  CheckOpenCL("clSetKernelArg");
  
  size_t local_size[3] = {8,8,8}; 
  size_t global_size[3] =
    { ((n[0] + local_size[0])/local_size[0])*local_size[0],
      ((n[1] + local_size[1])/local_size[1])*local_size[1],
      ((n[2] + local_size[2])/local_size[2])*local_size[2]
    };
  
  errcode = clEnqueueNDRangeKernel(CL.command_queue,
                kern, 3, NULL, global_size, local_size, 0, NULL, NULL);
  CheckOpenCL("clEnqueueNDRangeKernel");

  T.stop();
}
