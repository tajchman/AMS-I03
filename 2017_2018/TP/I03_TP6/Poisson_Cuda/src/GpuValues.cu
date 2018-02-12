#include "GpuValues.hxx"

GpuValues::GpuValues(const Parameters * prm) : Values(prm)
{
}

void GpuValues::allocate(size_t n) {
  deallocate();
  cudaMallocManaged(&m_u, n*sizeof(double));
}

void GPUValues::deallocate() {
  if (m_u != NULL) {
    cudaFree(m_u);
    m_u = NULL;
  }
}
