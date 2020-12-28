
__kernel void smoothGPU(__global float *g_odata,
			__global float *g_idata,
			int width, int height)
{
  const int idx = get_global_id (0);
  const int idy = get_global_id (1);
 
  if ((idx < width) && (idy < height)) {

    if ((idx >= nGauss) && (idx < width-nGauss) &&
	(idy >= nGauss) && (idy < height-nGauss)) {
      float sum = 0;
      for (int x = -nGauss; x <= nGauss; x++) { 
	for (int y = -nGauss; y <= nGauss; y++) { 
	  sum += d_GKernel[x+nGauss][y+nGauss]*g_idata[(idx+x)+(idy+y)*width];
	}
      }
      g_odata[idx + idy*width] = sum;
    }
    else
      g_odata[idx + idy*width] = g_idata[idx + idy*width];
  }
}
