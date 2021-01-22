#define CL_SILENCE_DEPRECATION

#include "OpenCL.hxx"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

OpenCL::OpenCL()
{
  platform_id = NULL;
  device_id = NULL;
  
  cl_int errcode;
  cl_uint uret;
  
  errcode = clGetPlatformIDs(1,
			     &platform_id,
			     &uret);
  CheckOpenCL("clGetPlatformIDs");
   
  errcode = clGetDeviceIDs(  platform_id,
			     CL_DEVICE_TYPE_DEFAULT,
			     1, 
			     &device_id,
			     &uret);
  CheckOpenCL("clGetDeviceIDs");

  context = clCreateContext
    (NULL, 1, &device_id, NULL, NULL, &errcode);
  CheckOpenCL("clCreateContext");

#if __OPENCL_VERSION__ > 120
  command_queue = clCreateCommandQueueWithProperties
    (context, device_id, CL_QUEUE_PROFILING_ENABLE, &errcode);
#else
  command_queue = clCreateCommandQueue
    (context, device_id, CL_QUEUE_PROFILING_ENABLE, &errcode);
#endif

  CheckOpenCL("clCreateCommandQueueWithProperties");
}

OpenCL::~OpenCL()
{
  clFlush(command_queue);
  clFinish(command_queue);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
}


cl_kernel OpenCL::new_kernel(const char *kernelName,
                             const char *fileName,
                             const char * header)
{
  cl_int errcode;

  std::string kernel_source = "";

  if (header)
    kernel_source = header;

  std::string fullName = INSTALL_PREFIX;
  fullName += "/";
  fullName += fileName;
  std::ifstream t(fullName.c_str());
  std::stringstream buffer;
  buffer << t.rdbuf();

  kernel_source += buffer.str();
  t.close();
//  std::cerr << "######" << std::endl << kernel_source << std::endl << "######" << std::endl;;

  const char * kernel_src = kernel_source.c_str();
  size_t kernel_len = kernel_source.size();

  cl_program program = clCreateProgramWithSource
    (context, 1, &kernel_src, &kernel_len, &errcode);
  CheckOpenCL("clCreateProgramWithSource");
  
  errcode = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (errcode < 0) {
    size_t len_log = 4999, ret_log;
    char log[len_log+1];
    clGetProgramBuildInfo (program,
                           device_id,
                           CL_PROGRAM_BUILD_LOG,
                           len_log,
                           log,
                           &ret_log);
    std::cerr << log << std::endl;
  }
  CheckOpenCL("clBuildProgram");
  
  // Create the OpenCL kernel
  cl_kernel k =  clCreateKernel(program, kernelName, &errcode);
  CheckOpenCL("clCreateKernel");
  return k;
  
}

void OpenCL::free_kernel(cl_kernel & k) {
    clReleaseKernel(k);
    k = NULL;
}

void OpenCL::info()
{
  cl_int ret, i;
  size_t param_value_size = 1023;
  char param_value[1024];
  size_t param_value_size_ret;
  cl_platform_info param_name;
  cl_ulong uvalue;
  size_t size[3];
  
  param_name = CL_PLATFORM_VERSION;
  ret = clGetPlatformInfo(platform_id,
			  param_name,
			  param_value_size,
			  param_value,
			  &param_value_size_ret);
  std::cerr << "\n\tOpenCL version : " << param_value << std::endl;

  param_name = CL_PLATFORM_NAME;
  ret = clGetPlatformInfo(platform_id,
			  param_name,
			  param_value_size,
			  param_value,
			  &param_value_size_ret);
  std::cerr << "\tPlatform name  : " << param_value << std::endl;
}
