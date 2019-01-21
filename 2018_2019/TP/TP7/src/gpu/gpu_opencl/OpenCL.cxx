#include "OpenCL.hxx"
#include <fstream>
#include <sstream>
#include <iostream>

OpenCL::OpenCL()
{
  platform_id = NULL;
  device_id = NULL;
  
  cl_int ret;
  cl_uint uret;
  
  ret = clGetPlatformIDs(1,
                         &platform_id,
                         &uret);
  
  ret = clGetDeviceIDs(  platform_id,
                         CL_DEVICE_TYPE_DEFAULT,
                         1, 
                         &device_id,
                         &uret);

  context = clCreateContext
    (NULL, 1, &device_id, NULL, NULL, &ret);
  command_queue = clCreateCommandQueueWithProperties
    (context, device_id, 0, &ret);
}

OpenCL::~OpenCL()
{
  clFlush(command_queue);
  clFinish(command_queue);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
}


cl_kernel OpenCL::new_kernel(const char *kernelName, const char *fileName)
{
  cl_int ret;

  std::ifstream t(fileName);
  std::stringstream buffer;
  buffer << t.rdbuf();

  std::string kernel_source = buffer.str();
  t.close();

  const char * kernel_src = kernel_source.c_str();
  size_t kernel_len = kernel_source.size();
  
  cl_program program = clCreateProgramWithSource
    (context, 1, &kernel_src, &kernel_len, &ret);
 
 clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  
  // Create the OpenCL kernel
  return clCreateKernel(program, kernelName, &ret);
  
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

  cl_device_id devices[10];
  cl_uint num_devices;
  clGetDeviceIDs(platform_id,
			CL_DEVICE_TYPE_ALL,
			10,
			devices,
			&num_devices);

  for (i=0; i<num_devices; i++) {
    param_name = CL_DEVICE_NAME;
    clGetDeviceInfo(devices[i],
		    param_name,
                    param_value_size,
                    param_value,
                    &param_value_size_ret);
    std::cerr << "\t\tDevice " << i << ":  " << param_value << std::endl;
  }
}
