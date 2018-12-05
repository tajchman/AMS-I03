
Calcul_GPU::Calcul_GPU(size_t n) : m_n(n)
{
  size_t bytes = m_n*sizeof(double);
  cudaMalloc(d_u, bytes);
  cudaMalloc(d_v, bytes);
  cudaMalloc(d_w, bytes);
}

Calcul_GPU::~Calcul_GPU()
{
  size_t bytes = n*sizeof(double);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
}

  
