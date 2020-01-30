#include "Network.h"
#include "MatrixVector.h"
#include <curand.h>
#include "timer.hxx"
#include <algorithm>
#include <random>
#include <iostream>

Network::Network(const std::vector<size_t>& sizes) :
  m_sizes(sizes),
  b0(m_sizes), biases(m_sizes),
  w0(m_sizes), weights(m_sizes),
  gen(2020)
{
}

void copy(std::vector<DataGPU> & t1,
               const std::vector<DataCPU> & t2)
{
    size_t i, n = t1.size();
    for (i = 0; i < n; i++) t1[i] = t2[i];
}

void Network::SGD(std::vector<DataCPU> &training_data_cpu,
                  int epochs, int mini_batch_size, double eta,
                  std::vector<DataCPU> &test_data_cpu)
{
  std::vector<DataGPU> training_data;
  copy(training_data, training_data_cpu);
  std::vector<DataGPU> test_data;
  copy(test_data, test_data_cpu);

  for (auto j = 0; j < epochs; j++) {
    Timer t;
    t.start();
    std::shuffle(training_data.begin(), training_data.end(), gen);
    t.stop();
    std::cerr << " shuffle  in " << t.elapsed() << " s" << std::endl;
    
    t.reinit();
    t.start();
    for (auto i = 0; i < training_data.size(); i += mini_batch_size) {
      auto iter = training_data.begin();
      std::advance(iter, i);
      update_mini_batch(iter, mini_batch_size, eta);
      std::cerr << "    " << i << "\r";
    }
    t.stop();
    std::cerr << " training in " << t.elapsed() << " s" << std::endl;
    
    if (test_data.size() != 0) {
      t.reinit();
      t.start();
      int ok = evaluate(test_data);
      t.stop();
      std::cerr << " testing  in " << t.elapsed() << " s" << std::endl;
      std::cout << "Epoch " << j
                << ": " << ok
                << " / " << test_data.size() << std::endl;
    }
    else
      std::cout << "Epoch " << j << " complete" << std::endl;
  }
}
// Update the network's weights and biases by applying
//	gradient descent using backpropagation to a single mini batch.
//	The "mini_batch" is a list of tuples "(x, y)", and "eta"
//	is the learning rate."""
void Network::update_mini_batch(DataGPUIterator td,
                       int mini_batch_size, double eta) {
  BiasesVector nabla_b(b0);
  WeightsVector nabla_w(w0);
   for (auto i = 0; i < mini_batch_size; ++i) {
     DataGPUIterator tdi = td + i;
     auto &x = tdi->first;  // test data
     auto &y = tdi->second; // expected result
     BiasesVector delta_nabla_b(b0);
     WeightsVector delta_nabla_w(w0);
     backprop(x, y, delta_nabla_b, delta_nabla_w);
    
     for (auto k = 0; k < biases.size(); ++k) {
       nabla_b[k] += delta_nabla_b[k];
       nabla_w[k] += delta_nabla_w[k];
     }
   }
//   for (auto i = 0; i < biases.size(); ++i) {
//     biases[i] -= eta / mini_batch_size * nabla_b[i];
//     weights[i] -= eta / mini_batch_size * nabla_w[i];
   }
 
// // Populates the gradient for the cost function
// // for the biases in the vector nabla_b
// // and the weights in nabla_w
 void Network::backprop(const Vector &x,
               const Vector &y,
               BiasesVector &nabla_b,
               WeightsVector &nabla_w) {
//   auto activation = x;
//   std::vector<Vector> activations; // Stores the activations of each layer
//   activations.push_back(x);
//   std::vector<Vector> zs; // The z vectors layer by layer
//   for (auto i = 0; i < biases.size(); ++i) {
//     ublas::vector<double> z = prod(weights[i], activation) + biases[i];
//     zs.push_back(z);
//     activation = z;
//     sigmoid(activation);
//     activations.push_back(activation);
//   }
//   // backward pass
//   auto iActivations = activations.end() - 1;
//   auto izs = zs.end() - 1;
//   sigmoid_prime(*izs);
//   Vector delta
//     = element_prod(cost_derivative(*iActivations, y), *izs);
//   auto ib = nabla_b.end() - 1;
//   auto iw = nabla_w.end() - 1;
//   *ib = delta;
//   iActivations--;
//   *iw = outer_prod(delta, trans(*iActivations));

//   auto iWeights = weights.end();
//   while (iActivations != activations.begin()) {
//     izs--;
//     iWeights--;
//     iActivations--;
//     ib--;
//     iw--;
//     sigmoid_prime(*izs);
//     delta = element_prod(prod(trans(*iWeights), delta), *izs);
//     *ib = delta;
//     *iw = outer_prod(delta, trans(*iActivations));
//  }
}

// Return the number of test inputs for which the neural
//	network outputs the correct result. Note that the neural
//	network's output is assumed to be the index of whichever
//	neuron in the final layer has the highest activation.


int Network::evaluate(const std::vector<DataGPU> &td) const {
  // return count_if
  //   (td.begin(), td.end(),
  //    [this](const DataGPU &testElement) {
  //     auto res = feedforward(testElement.first);
  //     return (std::distance(res.begin(),
  //                           max_element(res.begin(),
  //                                       res.end()))
  //             ==
  //             std::distance(testElement.second.begin(),
  //                           max_element(testElement.second.begin(),
  //                                       testElement.second.end())));
  //   });
  return 0;
}
  
// Return the vector of partial derivatives \partial C_x /
//	\partial a for the output activations.
// Vector cost_derivative
// (const Vector &output_activations,
//  const Vector) const {
//   return output_activations - y;
// }
//};

