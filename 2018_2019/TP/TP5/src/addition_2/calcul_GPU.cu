#include <iostream>
#include "calcul.hxx"
#include "timer.hxx"

__global__ void vecInit(double *a, double *b, int n)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  double x;
  
  if (id >= n) return;

  x = 1.0 * id;
  a[id] = sin(x)*sin(x);
  b[id] = cos(x)*cos(x);
}

__global__ void vecAdd(double *c, double *a, double *b, int n)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (id < n)
    c[id] = a[id] + b[id];
}


Calcul_GPU::Calcul_GPU(std::size_t n) : m_n(n)
{
  Timer T1; T1.start();
  
  std::size_t bytes = m_n*sizeof(double);
  cudaMalloc(&d_u, bytes);
  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_w, bytes);
    
  T1.stop();
  std::cerr << "\t\ttemps init 1 : " << T1.elapsed() << std::endl;
  Timer T2; T2.start();
  
  blockSize = 4096; 
  gridSize = (int)ceil((float)n/blockSize);
  
  vecInit<<<gridSize, blockSize>>>(d_u, d_v, n);

  cudaDeviceSynchronize();
  T2.stop();
  std::cerr << "\t\ttemps init 2 : " << T2.elapsed() << std::endl;
}

Calcul_GPU::~Calcul_GPU()
{
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
}

void Calcul_GPU::addition()
{
  Timer T; T.start();
  
  vecAdd<<<gridSize, blockSize>>>(d_w, d_u, d_v, m_n);
  
  cudaDeviceSynchronize();
  T.stop();
  std::cerr << "\t\ttemps add.   : " << T.elapsed() << std::endl;
}

double Calcul_GPU::verification()
{
  Timer T; T.start();
  
  std::size_t bytes = m_n*sizeof(double);
  std::vector<double> w(m_n);
  cudaMemcpy(w.data(), d_w, bytes, cudaMemcpyDeviceToHost);

  double s = 0;
  std::size_t i;
  for (i=0; i<m_n; i++)
    s = s + w[i];
  
  s = s/m_n - 1.0;
  
  T.stop();
  std::cerr << "\t\ttemps verif. : " << T.elapsed() << std::endl;

  return s;
}

  
