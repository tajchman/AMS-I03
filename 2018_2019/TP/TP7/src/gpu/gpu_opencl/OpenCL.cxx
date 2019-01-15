#include "OpenCL.hxx"
#include <fstream>
#include <sstream>

OpenCL::OpenCL()
{
  platform_id = NULL;
  device_id = NULL;
  
  cl_int ret;

  ret = clGetPlatformIDs(1,
                         &platform_id,
                         &ret_num_platforms);
  
  ret = clGetDeviceIDs(  platform_id,
                         CL_DEVICE_TYPE_DEFAULT,
                         1, 
                         &device_id,
                         &ret_num_devices);

  context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
}

cl_kernel =  OpenCL::new_kernel(const char *kernelName, const char *fileName)
{
  std::ifstream t(fileName);
  std::stringstream buffer;
  buffer << t.rdbuf();

  std::string kernel_source = buffer.str();
  t.close();

  const char * kernel_src = kernel_source.c_str();
  size_t kernel_len = kernel_source.size();
  
  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource
    (context, 1, &kernel_src, &kernel_len, &ret);
 
  // Build the program
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  
  // Create the OpenCL kernel
  return clCreateKernel(program, kernelName, &ret);
  
}


    
 

}
