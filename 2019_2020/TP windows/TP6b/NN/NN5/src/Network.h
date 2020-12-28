#pragma once
#include "MatrixVector.h"
#include <random>

class Network
{
public:
  Network(const std::vector<size_t>& sizes);
  
  using TrainingDataCPU = std::pair<std::vector<double>, std::vector<double>>;
  using TrainingDataCPUIterator =
    typename std::vector<TrainingDataCPU>::iterator;
  
  using TrainingDataGPU = std::pair<Vector, Vector>;
  using TrainingDataGPUIterator =
    typename std::vector<TrainingDataGPU>::iterator;
  
  void SGD(std::vector<TrainingDataCPU> &training_data,
           int epochs, int mini_batch_size, double eta,
           std::vector<TrainingDataCPU> &test_data);
    
private:
  
  std::vector<size_t> m_sizes;
  
  BiasesVector b0, biases;
  WeightsVector w0, weights;

  void update_mini_batch(TrainingDataGPUIterator td,
                         int mini_batch_size, double eta);

  void backprop(const Vector &x,
                const Vector &y,
                BiasesVector &nabla_b,
		WeightsVector &nabla_w);
  
  int evaluate(const std::vector<TrainingDataGPU> &td) const;
  std::random_device rd;
  std::mt19937 gen;

};
