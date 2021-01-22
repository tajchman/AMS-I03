
__kernel void InitOpenCL(
       __global double *u,
	 		 __global double *v,
			   int n)
{
  const int id = get_global_id (0);
  if (id >= n) return;

  double x = 1.0 * id;
  u[id] = sin(x)*sin(x);
  v[id] = cos(x)*cos(x);
}
