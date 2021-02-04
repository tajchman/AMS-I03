#define MAX_SOURCE_SIZE (0x100000)

#include "io_png.hxx"
#include <iostream>
#include <sstream>
#include <cmath>
#include "timerCL.hxx"
#include "OpenCL.hxx"
#include "cImageGPU.h"

void setGrey(cImageGPU &imageOut, const cImageGPU &imageIn,
	     cl_kernel kern, OpenCL & CL)
{
  TimerCL T(CL.command_queue);
  T.start();
  
  int n = imageIn.width * imageIn.height;
  int errcode;
  
  errcode = clSetKernelArg(kern, 0, sizeof(cl_mem), &imageOut.d_coef[0]);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 1, sizeof(cl_mem), &imageIn.d_coef[0]);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 2, sizeof(cl_mem), &imageIn.d_coef[1]);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 3, sizeof(cl_mem), &imageIn.d_coef[2]);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 4, sizeof(int), &n);
  CheckOpenCL("clSetKernelArg");
  
  size_t local_size = 256; 
  size_t global_size = ((n + local_size)/local_size)*local_size;
  
  errcode = clEnqueueNDRangeKernel(CL.command_queue,
                         kern, 1, NULL,
			 &global_size,
			 &local_size,
			 0, NULL, NULL);
  CheckOpenCL("clEnqueueNDRangeKernel");
  T.stop();
  std::cerr << "\t\tTime Gray image generation  "
	    << T.elapsed() << " s" << std::endl;  
}


#define nGauss 3
#define nGauss2 (2*nGauss+1)

std::string filterCreation() 
{ 
  float GKernel[nGauss2][nGauss2];
  double sigma = 1.0; 
    double g, r, s = 2.0 * sigma * sigma; 
  
    double sum = 0.0; 
  
    for (int x = -nGauss; x <= nGauss; x++) { 
        for (int y = -nGauss; y <= nGauss; y++) { 
            r = sqrt(x * x + y * y);
            g = (exp(-(r * r) / s)) / (M_PI * s);
            GKernel[x + nGauss][y + nGauss] = g; 
            sum += g; 
        } 
    } 
  
    for (int i = 0; i < nGauss2; ++i) 
        for (int j = 0; j < nGauss2; ++j) 
            GKernel[i][j] /= sum;

    std::ostringstream h;
    h << "\n\n__constant int nGauss = " << nGauss << ";\n";
    h << "\n\n__constant float d_GKernel[" <<nGauss2<< "][" <<nGauss2<< "] = {\n";
    for (int i = 0; i < nGauss2; ++i) {
      if (i > 0) h << "\n, ";
      h << "{";
      for (int j = 0; j < nGauss2; ++j) {
        if (j > 0) h << ", ";
        h << GKernel[i][j];
      }
      h << "}";
    }
    h << "};\n\n";
    return h.str();
}


void smooth (cImageGPU &imageOut, const cImageGPU &imageIn,
	     cl_kernel kern, OpenCL & CL)
{
  TimerCL T(CL.command_queue);
  T.start();
  
  int errcode;
  
  errcode = clSetKernelArg(kern, 0, sizeof(cl_mem), &imageOut.d_coef[0]);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 1, sizeof(cl_mem), &imageIn.d_coef[0]);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 2, sizeof(int), &imageIn.width);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 3, sizeof(int), &imageIn.height);
  CheckOpenCL("clSetKernelArg");
  
  size_t local_size[2] = {16, 16}; 
  size_t global_size[2] =
    { ((imageIn.width + local_size[0])/local_size[0])*local_size[0],
      ((imageIn.height + local_size[1])/local_size[1])*local_size[1]
    };
  
  errcode = clEnqueueNDRangeKernel(CL.command_queue,
                         kern, 2, NULL,
			 global_size,
			 local_size,
			 0, NULL, NULL);
  CheckOpenCL("clEnqueueNDRangeKernel");
  T.stop();
  std::cerr << "\t\tTime Smooth filter          "
	    << T.elapsed() << " s" << std::endl;  
}

void sobel (cImageGPU &imageOut, const cImageGPU &imageIn,
	     cl_kernel kern, OpenCL & CL)
{
  TimerCL T(CL.command_queue);
  T.start();
  
  int errcode;
  
  errcode = clSetKernelArg(kern, 0, sizeof(cl_mem), &imageOut.d_coef[0]);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 1, sizeof(cl_mem), &imageIn.d_coef[0]);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 2, sizeof(int), &imageIn.width);
  CheckOpenCL("clSetKernelArg");
  errcode = clSetKernelArg(kern, 3, sizeof(int), &imageIn.height);
  CheckOpenCL("clSetKernelArg");
  
  size_t local_size[2] = {16, 16}; 
  size_t global_size[2] =
    { ((imageIn.width + local_size[0])/local_size[0])*local_size[0],
      ((imageIn.height + local_size[1])/local_size[1])*local_size[1]
    };
  
  errcode = clEnqueueNDRangeKernel(CL.command_queue,
                         kern, 2, NULL,
			 global_size,
			 local_size,
			 0, NULL, NULL);
  CheckOpenCL("clEnqueueNDRangeKernel");
  T.stop();
  std::cerr << "\t\tTime Sobel filter          "
	    << T.elapsed() << " s" << std::endl;  
}

void process(cImage &imageOut, const cImage &imageIn)
{ 
  Timer T;
  T.start();
  
  int w = imageIn.width,
    h = imageIn.height;
  
  OpenCL CL;
//   CL.info();
   
  cImageGPU imageTemp0(imageIn, CL);
  cImageGPU imageTemp1(w, h, 1, CL);
  cImageGPU imageTemp2(w, h, 1, CL);
  cImageGPU imageTemp3(w, h, 1, CL);
  
  T.stop();
  std::cerr << "\n\tTime send to GPU           "
	    << T.elapsed() << " s" << std::endl;

  T.reinit();
  T.start();
  
  cl_kernel grayKernel = CL.new_kernel("setGreyGPU", "gray.cl");

  std::string smooth_header = filterCreation();
  cl_kernel smoothKernel = CL.new_kernel("smoothGPU", "smooth.cl",
                                         smooth_header.c_str());
  
  cl_kernel sobelKernel = CL.new_kernel("sobelGPU", "sobel.cl");
  
  T.stop();
  std::cerr << "\n\tTime compile kernels      "
	    << T.elapsed() << " s" << std::endl;

  std::cerr << std::endl;
  
  TimerCL T_compute(CL.command_queue);
  T_compute.start();

  setGrey(imageTemp1, imageTemp0, grayKernel,   CL);
  smooth (imageTemp2, imageTemp1, smoothKernel, CL);
  sobel  (imageTemp3, imageTemp2, sobelKernel,  CL);

  
  T_compute.stop();
  std::cerr << "\n\tTime compute on GPU        "
	    << T_compute.elapsed() << " s" << std::endl;

  T.restart();
  imageOut = cImage(imageTemp3);
  
  T.stop();
  std::cerr << "\tTime get from GPU          "
	    << T.elapsed() << " s" << std::endl;
  
  CL.free_kernel(grayKernel);
  CL.free_kernel(smoothKernel);

  
  // cl_kernel smoothKernel = CL.new_kernel("smooth", "smooth.cl");
  // cl_kernel sobelKernel = CL.new_kernel("sobel", "sobel.cl");
}
