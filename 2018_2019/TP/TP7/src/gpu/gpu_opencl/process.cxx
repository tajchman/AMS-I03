#define MAX_SOURCE_SIZE (0x100000)

#include "io_png.hxx"
#include <iostream>
#include <cmath>
#include "timer.hxx"
#include "OpenCL.hxx"
#include "cImageGPU.h"

void setGrey(cImageGPU &imageOut, const cImageGPU &imageIn,
	     cl_kernel kern, OpenCL & CL)
{
  Timer T;
  T.start();

  clSetKernelArg(kern, 0, sizeof(cl_mem), &imageOut.d_coef[0]);
  clSetKernelArg(kern, 1, sizeof(cl_mem), &imageIn.d_coef[0]);
  clSetKernelArg(kern, 2, sizeof(cl_mem), &imageIn.d_coef[1]);
  clSetKernelArg(kern, 3, sizeof(cl_mem), &imageIn.d_coef[2]);

  size_t global_size[2] = { imageIn.width, imageIn.height };
  size_t local_size[2] = {16, 16}; 
  
  clEnqueueNDRangeKernel(CL.command_queue,
                         kern, 2, NULL,
			 global_size,
			 local_size,
			 0, NULL, NULL);
  T.stop();
  std::cerr << "\t\tTime Gray image generation "
	    << T.elapsed() << " s" << std::endl;  
}

/*
#define nGauss 3
#define nGauss2 (2*nGauss+1)

void filterCreation(float GKernel[][nGauss2]) 
{ 
    double sigma = 1.0; 
    double g, r, s = 2.0 * sigma * sigma; 
  
    double sum = 0.0; 
  
    // generating nGauss2xnGauss2 kernel 

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
}

void smooth(cImage &imageOut, const cImage &imageIn)
{
  Timer T;
  T.start();

  float GKernel[nGauss2][nGauss2];
  filterCreation(GKernel);
    
  int i, j, k, c;
  
  imageOut = imageIn;

#pragma omp parallel for private(i,j)
  for (i=nGauss; i<imageIn.width-nGauss; i++) {
    for (j=nGauss; j<imageIn.height-nGauss; j++) {
        float sum = 0;
        for (int x = -nGauss; x <= nGauss; x++) { 
          for (int y = -nGauss; y <= nGauss; y++) { 
            sum += GKernel[x+nGauss][y+nGauss]*imageIn(i+x, j+y, 0);
          }
        }
        imageOut(i,j,0) = sum;
      }
    }

  T.stop();
  std::cerr << "\t\tTime Gauss smoothing       " << T.elapsed() << " s" << std::endl;  
 }

void setGrey(cImage &imageOut, const cImage &imageIn)
{
  Timer T;
  T.start();

  int i,j;

  if (imageIn.ncolors == 3) {
    imageOut.resize(imageIn.width, imageIn.height, 1);
    
#pragma omp parallel for private(i,j)
    for (i=0; i<imageIn.width; i++)
      for (j=0; j<imageIn.height; j++) {
        imageOut(i,j,0)
          = 0.21*imageIn(i,j,0)
          + 0.72*imageIn(i,j,1)
          + 0.07*imageIn(i,j,2);
        }
  }
  else
    imageOut = imageIn;

  T.stop();
  std::cerr << "\t\tTime Gray image generation " << T.elapsed() << " s" << std::endl;  
}

void sobel(cImage &imageOut, const cImage &imageIn)
{
  Timer T;
  T.start();

  int i,j,m,n;
  float s;
  int dx[3][3] = {{ 1, 0,-1},{ 2, 0,-2},{ 1, 0,-1}};
  int dy[3][3] = {{ 1, 2, 1},{ 0, 0, 0},{-1,-2,-1}};
  int sum;
    
  imageOut.resize(imageIn.width, imageIn.height, 1);
  
#pragma omp parallel for private(i,j, m,n,s,sum)
  for (i=1; i<imageIn.width-1; i++) {
    for (j=1; j<imageIn.height-1; j++) {
        int sumx=0;
        int sumy=0;
        for(m=-1; m<=1; m++)
          for(n=-1; n<=1; n++) {
            s = (int) imageIn(i+m,j+n,0);
            sumx+=s*dx[m+1][n+1];
            sumy+=s*dy[m+1][n+1];
          }
        sum=abs(sumx)+abs(sumy);
       	imageOut(i,j,0) = (sum>255) ? 255 : sum;
    }
  }

  T.stop();
  std::cerr << "\t\tTime Sobel filter          " << T.elapsed() << " s" << std::endl;  
}
*/

void process(cImage &imageOut, const cImage &imageIn)
{ 
  Timer T;
  T.start();
  
  int w = imageIn.width,
    h = imageIn.height;
  
  OpenCL CL;
   CL.info();
   
  cImageGPU imageTemp0(imageIn, CL);
  cImageGPU imageTemp1(w, h, 1, CL);
  cImageGPU imageTemp2(w, h, 1, CL);
  cImageGPU imageTemp3(w, h, 1, CL);

  cl_kernel grayKernel = CL.new_kernel("setGreyGPU", "gray.cl");
  setGrey(imageTemp3, imageTemp0, grayKernel, CL);

  imageOut = cImage(imageTemp3);
  
  CL.free_kernel(grayKernel);

  
  // cl_kernel smoothKernel = CL.new_kernel("smooth", "smooth.cl");
  // cl_kernel sobelKernel = CL.new_kernel("sobel", "sobel.cl");
}
