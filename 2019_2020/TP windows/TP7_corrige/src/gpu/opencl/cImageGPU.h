#include "io_png.hxx"
#include <iostream>
#include <cmath>
#include "OpenCL.hxx"

inline cl_mem * reserveCoefs(int w, int h, int nc, cl_context context)
{
  cl_mem * d = (cl_mem *) malloc(sizeof(cl_mem) * nc);
  size_t bytes = w*h*sizeof(float);
  cl_int errcode;
  
  for (int c=0; c<nc; c++) {
    d[c] = clCreateBuffer (context, CL_MEM_READ_WRITE,
			   bytes, NULL, &errcode);
    CheckOpenCL("clCreateBuffer");
  }

  return d;
}

class cImageGPU {
 public:

 cImageGPU(int w, int h, int nc, OpenCL & CL) :
  height(h),
    width(w),
    ncolors(nc),
    imageSize(h*w),
    d_CL(CL) {
      bytes = imageSize*sizeof(float);
      d_coef = reserveCoefs(width, height, ncolors, d_CL.context);
    }
  
 cImageGPU(const cImage &I, OpenCL & CL) :
  height(I.height),
    width(I.width),
    ncolors(I.ncolors),
    imageSize(height*width),
    d_CL(CL) {

      d_coef = reserveCoefs(width, height, ncolors, d_CL.context);
      int i,j,k,c;
      bytes = imageSize*sizeof(float);
      float * p = (float *) malloc(bytes);
      
      cl_int errcode;
      for (c=0; c<I.ncolors; c++) {
	for (j=0, k=0; j<height; j++)
	  for (i=0; i<width; i++, k++)
	    p[k] = I(i,j,c);
	
	errcode = clEnqueueWriteBuffer(d_CL.command_queue, d_coef[c], CL_TRUE,
				       0, bytes, p,
				       0, NULL, NULL);
	CheckOpenCL("clEnqueueWriteBuffer");
      }
      free(p);
    }
  
  operator cImage() const {
    cImage I(width, height, ncolors);
    
    int i,j,k,c;
    float * p = (float *) malloc(bytes);

    for (c=0; c<I.ncolors; c++) {
      clEnqueueReadBuffer(d_CL.command_queue,
                          d_coef[c], CL_TRUE,
			       0, bytes, p,
			       0, NULL, NULL);
      for (j=0,k=0; j<height; j++)
	for (i=0; i<width; i++, k++)
	  I(i,j,c) = p[k];
    }
    free(p);
    return I;
  }

  ~cImageGPU() {
    int c;
    for (c=0; c<ncolors; c++)
      clReleaseMemObject(d_coef[c]);
    free(d_coef);
  }
  
  size_t height, width, ncolors, bytes, imageSize;
  cl_mem * d_coef;
  OpenCL & d_CL;
};

