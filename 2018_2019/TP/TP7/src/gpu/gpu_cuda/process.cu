#include "operation.h"

#include "io_png.hxx"
#include <iostream>
#include <cmath>
#include "timer.hxx"
#include "cuda_check.cuh"

const int  blockSize = 512;

float ** reserveCoefs(int w, int h, int nc)
{
    float ** d = (float **) malloc(sizeof(float *) * nc);
    size_t bytes = w*h*sizeof(float) * nc;

    float * p;
    CUDA_CHECK_OP(cudaMalloc(&p, bytes));
    for (int c=0; c<nc; c++)
      d[c] = p + w*h;

    return d;
}

class cImageGPU {
 public:

  cImageGPU(int w, int h, int nc) :
    height(h),
    width(w),
    ncolors(nc),
    frameSize(h*wb),
    bytes(frameSize*sizeof(float) {
    d_coef = reserveCoefs(w, h, nc);
  }
  
  cImageGPU(const cImage &I) : height(I.height),
    width(I.width),
    ncolors(I.ncolors) {

    d_coef = reserveCoefs(width, height, ncolors);
    int imageSize = width*height;
    size_t bytes = ncolors*imageSize*sizeof(float);
    float * p0 = (float *) malloc(bytes);

    for (c=0; c<I.ncolors; c++) {
      float * p = p0 + imageSize;
      for (int i=0; i<imageSize; i++)
	p[i] = I.coef[i*I.ncolors + c];
    }

    cudaMemcpy(d_coef[0], p0, bytes, cudaMemcpyHostToDevice);
    
    free(h0);
  }

  operator cImage() {
    cImage I;
    I.resize(width, height, ncolors);
    cudaMemcpy(I.coef.data(), d_coef, bytes, cudaMemcpyDeviceToHost);
    size_t bytes = width*height*ncolors*sizeof(float);
    float * h = (float *) malloc(bytes);
    cudaMemcpy(h, d_coef[0], bytes, cudaMemcpyDeviceToHost);

    for (c=0; c<I.ncolors; c++) {
      for (int i=0; i<w*h; i++)
	I.coef[i*I.ncolors + c] = h[i];
    }
    free(h);
    return I;
  }
  
  int height, width, ncolors, bytes, frameSize;
  float ** d_coef;
};

#define nGauss 3
#define nGauss2 (2*nGauss+1)

void filterCreation(float GKernel[][nGauss2]) 
{ 
    float sigma = 1.0; 
    float g, r, s = 2.0 * sigma * sigma; 
  
    float sum = 0.0; 
  
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

void smooth(cImageGPU &imageOut, const cImageGPU &imageIn)
{
  Timer T;
  T.start();

  // float GKernel[nGauss2][nGauss2];
  // filterCreation(GKernel);
    
  // int i, j;
  
  // imageOut = imageIn;
  // for (i=nGauss; i<imageIn.width-nGauss; i++) {
  //   for (j=nGauss; j<imageIn.height-nGauss; j++) {
  //       float sum = 0;
  //       for (int x = -nGauss; x <= nGauss; x++) { 
  //         for (int y = -nGauss; y <= nGauss; y++) { 
  //           sum += GKernel[x+nGauss][y+nGauss]*imageIn(i+x, j+y, 0);
  //         }
  //       }
  //       imageOut(i,j,0) = sum;
  //     }
  //   }

  T.stop();
  std::cerr << "\t\tTime Gauss smoothing       " << T.elapsed() << " s" << std::endl;  
 }

__global__ void setGreyGPU(float *g_odata,
			   float *g_ired,
			   float *g_igreen,
			   float *g_iblue,
			   size_t n)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (id < n)
    g_odata[id]
      = 0.21*g_ired[id]
      + 0.72*g_igreen[id]
      + 0.07*g_iblue[id];
}


void setGrey(cImageGPU &imageOut, const cImageGPU &imageIn)
{
  Timer T;
  T.start();

  int n = imageIn.width * imageOut.height;
  int gridSize = (int)ceil((double)n/blockSize);
  setGreyGPU<<<gridSize, blockSize>>>(imageOut.d_coef[0],
				      imageIn.d_coef[0],
				      imageIn.d_coef[1],
				      imageIn.d_coef[2],
				      n);
  CUDA_CHECK_KERNEL();

  T.stop();
  std::cerr << "\t\tTime Gray image generation " << T.elapsed() << " s" << std::endl;  
}

void sobel(cImageGPU &imageOut, const cImageGPU &imageIn)
{
  Timer T;
  T.start();

  // int i,j,m,n;
  // float s;
  // int dx[3][3] = {{ 1, 0,-1},{ 2, 0,-2},{ 1, 0,-1}};
  // int dy[3][3] = {{ 1, 2, 1},{ 0, 0, 0},{-1,-2,-1}};
  // int sum, sumx, sumy;
    
  // imageOut.resize(imageIn.width, imageIn.height, 1);
  
  // for (i=1; i<imageIn.width-1; i++) {
  //   for (j=1; j<imageIn.height-1; j++) {
  //       sumx=0;
  //       sumy=0;
  //       for(m=-1; m<=1; m++)
  //         for(n=-1; n<=1; n++) {
  //           s = (int) imageIn(i+m,j+n,0);
  //           sumx+=s*dx[m+1][n+1];
  //           sumy+=s*dy[m+1][n+1];
  //         }
  //       sum=abs(sumx)+abs(sumy);
  //      	imageOut(i,j,0) = (sum>255) ? 255 : sum;
  //   }
  // }

  T.stop();
  std::cerr << "\t\tTime Sobel filter          " << T.elapsed() << " s" << std::endl;  
}

void process(cImage &imageOut, const cImage &imageIn)
{
  Timer T;
  T.start();
  
  int w = imageIn.width,
    h = imageIn.height;
  
  cImageGPU
    imageTemp0(imageIn),
    imageTemp1(w, h, 1),
    imageTemp2(w, h, 1),
    imageTemp3(w, h, 1);

  T.stop();
  std::cerr << "\t\tTime send to GPU " << T.elapsed() << " s" << std::endl;
  
  std::cerr << std::endl;  
  setGrey(imageTemp3, imageTemp0);  
  // smooth (imageTemp2, imageTemp1);
  // sobel  (imageTemp3, imageTemp2);

  T.restart();
  imageOut = cImage(imageTemp3);
  T.stop();
  std::cerr << "\t\tTime get from GPU " << T.elapsed() << " s" << std::endl;
}
