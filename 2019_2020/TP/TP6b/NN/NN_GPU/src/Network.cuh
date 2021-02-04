#pragma once
#include "MatrixVectorCUDA.cuh"

class NetworkCUDA
{
public:
  NetworkCUDA(const std::vector<size_t>& sizes);
  
private:
  
  std::vector<size_t> m_sizes;
  
  BiasesVector b0, biases;
  WeightsVector w0, weights;

};
