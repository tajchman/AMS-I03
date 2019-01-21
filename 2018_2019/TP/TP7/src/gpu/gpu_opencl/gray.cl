
__kernel void setGreyGPU(__global float *g_odata,
	 		 __global float *g_ired,
			 __global float *g_igreen,
			 __global float *g_iblue,
			   size_t n)
{
  const int id = get_global_id (0);
  
  if (id < n)
    g_odata[id]
      = 0.21*g_ired[id]
      + 0.72*g_igreen[id]
      + 0.07*g_iblue[id];
}
