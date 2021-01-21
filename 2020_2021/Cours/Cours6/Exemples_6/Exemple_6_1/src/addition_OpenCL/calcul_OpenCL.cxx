#include <iostream>
#include "calcul_OpenCL.hxx"
#include "timer.hxx"

inline cl_mem * reserve(int w, int h, int nc, cl_context context)
{
  cl_mem * d = (cl_mem *) malloc(sizeof(cl_mem) * nc);
  size_t bytes = w*h*sizeof(double);
  cl_int errcode;
  
  for (int c=0; c<nc; c++) {
    d[c] = clCreateBuffer (context, CL_MEM_READ_WRITE,
			   bytes, NULL, &errcode);
    CheckOpenCL("clCreateBuffer");
  }

  return d;
}

Calcul_GPU::Calcul_GPU(int m) : n(m)
{
  Timer & T = GetTimer(T_AllocId); T.start();
  
  int bytes = n*sizeof(double);


  T.stop();
}

Calcul_GPU::~Calcul_GPU()
{
  Timer & T = GetTimer(T_FreeId); T.start();

  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);

  T.stop();
}

__global__ void vecInit(double *a, double *b, int n)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  double x;
  
  if (id >= n) return;

  x = 1.0 * id;
  a[id] = sin(x)*sin(x);
  b[id] = cos(x)*cos(x);
}

void Calcul_GPU::init()
{
  Timer & T = GetTimer(T_InitId); T.start();

  blockSize = 512;
  gridSize = (unsigned int) ceil((double)n/blockSize);
  
  vecInit<<<gridSize, blockSize>>>(d_u, d_v, n);
  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();

  T.stop();
}

__global__ void vecAdd(double *c, double *a, double *b, int n)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (id < n)
    c[id] = a[id] + b[id];
}

void Calcul_GPU::addition()
{
  Timer & T = GetTimer(T_AddId); T.start();
  
  vecAdd<<<gridSize, blockSize>>>(d_w, d_u, d_v, n);
  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();

  T.stop();
}

double Calcul_GPU::verification()
{
  Timer & T1 = GetTimer(T_CopyId);
  T1.start();
  
  int bytes = n*sizeof(double);
  std::vector<double> w(n);
  cudaMemcpy(w.data(), d_w, bytes, cudaMemcpyDeviceToHost);

  T1.stop();

  Timer & T = GetTimer(T_VerifId);
  T.start();

  double s = 0;
  for (int i=0; i<n; i++)
    s = s + w[i];
  
  s = s/n - 1.0;
  
  T.stop();

  return s;
}

