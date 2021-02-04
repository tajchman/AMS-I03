#pragma once
#include "MatrixVector.h"
#include <random>

class Network
{
public:
  Network(const std::vector<size_t>& sizes);
  
  void SGD(std::vector<DataCPU> &training_data,
           int epochs, int mini_batch_size, double eta,
           std::vector<DataCPU> &test_data);
    
private:
  
  std::vector<size_t> m_sizes;
  
  BiasesVector b0, biases;
  WeightsVector w0, weights;

  void update_mini_batch(DataGPUIterator td,
                         int mini_batch_size, double eta);

  void backprop(const Vector &x,
                const Vector &y,
                BiasesVector &nabla_b,
		        WeightsVector &nabla_w);
  
  int evaluate(const std::vector<DataGPU> &td) const;
  std::random_device rd;
  std::mt19937 gen;

};
