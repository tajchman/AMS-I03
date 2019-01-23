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

  command_queue = clCreateCommandQueueWithProperties
    (context, device_id, 0, &errcode);
  CheckOpenCL("clCreateCommandQueueWithProperties");
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
  cl_int errcode;

  std::string fullName = INSTALL_PREFIX;
  fullName += "/";
  fullName += fileName;
  std::ifstream t(fullName);
  std::stringstream buffer;
  buffer << t.rdbuf();

  std::string kernel_source = buffer.str();
  t.close();

  const char * kernel_src = kernel_source.c_str();
  size_t kernel_len = kernel_source.size();

  cl_program program = clCreateProgramWithSource
    (context, 1, &kernel_src, &kernel_len, &errcode);
  CheckOpenCL("clCreateProgramWithSource");
  
  errcode = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
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

  cl_device_id devices[10];
  cl_uint num_devices;
  clGetDeviceIDs(platform_id,
			CL_DEVICE_TYPE_ALL,
			10,
			devices,
			&num_devices);

  for (i=0; i<num_devices; i++) {
    std::cerr << "\tDevice " << i << std::endl;

    param_name = CL_DEVICE_NAME;
    clGetDeviceInfo(devices[i],
		    param_name,
                    param_value_size,
                    param_value,
                    &param_value_size_ret);
    std::cerr << "\t\tName               : "
	      << param_value << std::endl;
    
    param_name =  CL_DEVICE_GLOBAL_MEM_SIZE;
    clGetDeviceInfo(devices[i],
		    param_name,
                    sizeof(uvalue),
                    &uvalue,
                    &param_value_size_ret);
    std::cerr << "\t\tGlobal memory size = "
	      << uvalue << std::endl;
    
    param_name =  CL_DEVICE_MAX_WORK_GROUP_SIZE;
    clGetDeviceInfo(devices[i],
		    param_name,
                    sizeof(size_t),
                    size,
                    &param_value_size_ret);
    std::cerr << "\t\tMax. work group size = "
	      << size[0] << std::endl;
    
    param_name = CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS;
    clGetDeviceInfo(devices[i],
		    param_name,
                    sizeof(uvalue),
                    &uvalue,
                    &param_value_size_ret);
    
    param_name = CL_DEVICE_MAX_WORK_ITEM_SIZES;
    clGetDeviceInfo(devices[i],
		    param_name,
                    sizeof(size_t) * uvalue,
                    size,
                    &param_value_size_ret);
    std::cerr << "\t\tMax. work item sizes = (";
    for (cl_ulong l=0; l<uvalue; l++)  {
      std::cerr << size[l];
      if (l<uvalue-1) std::cerr << ", ";
    }
    std::cerr << ")" << std::endl;
  }
}
