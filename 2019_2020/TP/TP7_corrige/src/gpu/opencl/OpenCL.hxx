#ifndef __OPENCL_HXX__
#define __OPENCL_HXX__

#include <cstdlib>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>
#endif

#define CheckOpenCL(function)                   \
  if (errcode != CL_SUCCESS) {			\
    std::cerr << __FILE__                       \
              << " (" << __LINE__ << ") : "	\
              << function << " : error code "	\
              << errcode << std::endl;          \
      std::exit(errcode);			\
  }

class OpenCL
{
public:
  OpenCL();
  ~OpenCL();
  
  cl_kernel new_kernel (const char * kernelName,
			const char * fileName,
                        const char * header = NULL);
  void free_kernel(cl_kernel &k);
  
  cl_mem    new_memobj(size_t s);

  void      free_memobj(cl_mem);

  void info();
  
  cl_platform_id platform_id;
  cl_device_id device_id;   
  cl_context context;
  cl_command_queue command_queue;

};

#endif
