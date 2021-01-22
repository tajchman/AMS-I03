#include "io_png.hxx"
#include <iostream>
#include <cmath>

inline double ** reserveCoefs(int w, int h, int nc)
{
  double ** d = (double **) malloc(sizeof(double *) * nc);
  size_t bytes = w*h*sizeof(double);

  for (int c=0; c<nc; c++)
    CUDA_CHECK_OP(cudaMalloc(&(d[c]), bytes));

  return d;
}

class cImageGPU {
 public:

 cImageGPU(int w, int h, int nc) :
  height(h),
    width(w),
    ncolors(nc),
    imageSize(h*w),
    d_coef(reserveCoefs(w, h, nc)) {
    bytes = imageSize*sizeof(double);
  }
  
 cImageGPU(const cImage &I) :
  height(I.height),
    width(I.width),
    ncolors(I.ncolors),
    imageSize(height*width),
    d_coef(reserveCoefs(width, height, ncolors)) {
	
	int i,j,k,c;
	bytes = imageSize*sizeof(double);
	double * p = (double *) malloc(bytes);
	
	for (c=0; c<I.ncolors; c++) {
	  for (j=0, k=0; j<height; j++)
	    for (i=0; i<width; i++, k++)
	      p[k] = I(i,j,c);
	  cudaMemcpy(d_coef[c], p, bytes, cudaMemcpyHostToDevice);
	}
	free(p);
      }
  
  operator cImage() const {
    cImage I(width, height, ncolors);
    
    int i,j,k,c;
    double * p = (double *) malloc(bytes);

    for (c=0; c<I.ncolors; c++) {
      cudaMemcpy(p, d_coef[c], bytes, cudaMemcpyDeviceToHost);
      for (j=0,k=0; j<height; j++)
	for (i=0; i<width; i++, k++)
	  I(i,j,c) = (float) p[k];
    }
    free(p);
    return I;
  }
  
  int height, width, ncolors, bytes, imageSize;
  double ** d_coef;
};
