#include "gpu_values.hxx"

void GPUValues::allocate(size_t n) {
	deallocate();
	cudaMallocManaged(&m_u, n*sizeof(double));
}

void GPUValues::deallocate() {
	if (m_u != NULL) {
		cudaFree(m_u);
		m_u = NULL;
	}
}
