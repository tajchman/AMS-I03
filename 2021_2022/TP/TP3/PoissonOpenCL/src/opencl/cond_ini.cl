__kernel void sobelGPU(__global float * gOut,
                       __global float * gIn,
		       int width, int height)
{
}

__device__
double cond_ini(double x, double y, double z)
{
  return 0.0;
}

__device__
double cond_lim(double x, double y, double z)
{
  return 2.0;
}

__device__
double force(double x, double y, double z)
{
  return x*(1-x) * y*(1-y) * z*(1-z);
}
