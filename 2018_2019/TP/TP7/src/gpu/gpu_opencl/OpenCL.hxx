#ifndef __OPENCL_HXX__
#define __OPENCL_HXX__

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

class OpenCL
{
public:
  OpenCL();

  cl_kernel new_kernel(const char * kernelName,
                       const char * fileName);
  cl_mem    new_memobj(size_t s);

  void      free_memobj(cl_mem);

private:
  cl_platform_id platform_id;
  cl_device_id device_id;   
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_context context;
  cl_command_queue command_queue;

};

#endif
