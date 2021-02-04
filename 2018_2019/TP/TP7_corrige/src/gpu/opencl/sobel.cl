__constant int d_dx[3][3] = {{ 1, 0,-1},{ 2, 0,-2},{ 1, 0,-1}};
__constant int d_dy[3][3] = {{ 1, 2, 1},{ 0, 0, 0},{-1,-2,-1}};

__kernel void sobelGPU(__global float * gOut,
                       __global float * gIn,
		       int width, int height)
{
  const int idx = get_global_id (0);
  const int idy = get_global_id (1);
  
  if ((idx < width) && (idy < height)) {

    if ((idx > 0) && (idx < width-1) &&
	(idy > 0) && (idy < height-1)) {
      int x,y;
      float s;
      float sum, sumx = 0, sumy = 0;
//      printf("%d %d : %f\n", idx, idy, gIn[idx + idy*width] );

      for(x=-1; x<=1; x++)
	for(y=-1; y<=1; y++) {
	  s = gIn[(idx+x)+(idy+y)*width];
	  sumx+=s*d_dx[x+1][y+1];
	  sumy+=s*d_dy[x+1][y+1];
	}
      sum=fabs(sumx)+fabs(sumy);
      gOut[idx + idy*width] = (sum>255.0) ? 255.0 : sum;
//      printf("%d %d : %f\n", idx, idy, sum);
    }
    else
      gOut[idx + idy*width] = gIn[idx + idy*width];
  }
}
