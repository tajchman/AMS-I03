
__kernel void AddOpenCL(
       __global double *w,
       __global double *u,
	 		 __global double *v,
			   int n)
{
  const int id = get_global_id (0);
  if (id >= n) return;

  w[id] = u[id] + v[id];
}
