#ifndef _NETWORK_HXX_
#define _NETWORK_HXX_

#include <random>
#include <vector>
#include "timer.hxx"

using namespace boost::numeric;

// Randomize as ublas vector
void Randomize(ublas::vector<double> &vec, std::mt19937 &gen) {
  std::normal_distribution<> d(0, 1);
  for (auto &e : vec) {
    e = d(gen);
  }
}

// Randomize as ublas matrix
void Randomize(ublas::matrix<double> &m, std::mt19937 &gen) {
  std::normal_distribution<> d(0, 1);
  for (auto &e : m.data()) {
    e = d(gen);
  }
}

// The sigmoid function.
void sigmoid(ublas::vector<double> &v) {
  for (auto &iz : v) {
    iz = 1.0 / (1.0 + exp(-iz));
  }
}
// Derivative of the sigmoid function.
void sigmoid_prime(ublas::vector<double> &v) {
  for (auto &iz : v) {
    iz = 1.0 / (1.0 + exp(-iz));
    iz = iz * (1 - iz);
  }
}

class Network {
private:
  std::vector<int> m_sizes;
  using BiasesVector = std::vector<ublas::vector<double>>;
  using WeightsVector = std::vector<ublas::matrix<double>>;
  BiasesVector b0, biases;
  WeightsVector w0, weights;
  std::random_device rd;
  std::mt19937 gen;
  
public:
  // The vector<int> sizes contains the number of neurons in the
  //	respective layers of the network.For example, if the list
  //	was{ 2, 3, 1} then it would be a three - layer network, with the
  //	first layer containing 2 neurons, the second layer 3 neurons,
  //	and the third layer 1 neuron.The biases and weights for the
  //	network are initialized randomly, using a Gaussian
  //	distribution with mean 0, and variance 1.  Note that the first
  //	layer is assumed to be an input layer, and by convention we
  //	won't set any biases for those neurons, since biases are only
  //	ever used in computing the outputs from later layers.
  Network(const std::vector<int> &sizes) : m_sizes(sizes), gen(2020) /*gen(rd())*/ {

    PopulateZeroWeightsAndBiases(b0, w0);
    biases = b0;
    weights = w0;
    for (auto &b : biases)
      Randomize(b, gen);
    for (auto &w : weights)
      Randomize(w, gen);
  }
  
  // Initialise the array of Biases and Matrix of weights
  void PopulateZeroWeightsAndBiases(BiasesVector &b, WeightsVector &w) const {
    for (size_t i = 1; i < m_sizes.size(); ++i) {
      b.push_back(ublas::zero_vector<double>(m_sizes[i]));
      w.push_back(ublas::zero_matrix<double>(m_sizes[i], m_sizes[i - 1]));
    }
  }
  // Returns the output of the network if the input is a
  ublas::vector<double> feedforward(ublas::vector<double> a) const {
    for (auto i = 0; i < biases.size(); ++i) {
      ublas::vector<double> c = prod(weights[i], a) + biases[i];
      sigmoid(c);
      a = c;
    }
    return a;
  }
  // Type definition of the Training data
  using TrainingData = std::pair<ublas::vector<double>, ublas::vector<double>>;
  using TrainingDataIterator = typename std::vector<TrainingData>::iterator;
  //	Train the neural network using mini-batch stochastic
  //	gradient descent.The training_data is a vector of pairs
  // representing the training inputs and the desired
  //	outputs.The other non - optional parameters are
  //	self - explanatory.If test_data is provided then the
  //	network will be evaluated against the test data after each
  //	epoch, and partial progress printed out.This is useful for
  //	tracking progress, but slows things down substantially.
  void SGD(std::vector<TrainingData> training_data,
           int epochs, int mini_batch_size, double eta,
           std::vector<TrainingData> test_data)
  {
    for (auto j = 0; j < epochs; j++) {
      Timer t;
      t.start();
      //      std::shuffle(training_data.begin(), training_data.end(), rd);
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
  void update_mini_batch(TrainingDataIterator td,
                         int mini_batch_size, double eta) {
    std::vector<ublas::vector<double>> nabla_b(b0);
    std::vector<ublas::matrix<double>> nabla_w(w0);
    for (auto i = 0; i < mini_batch_size; ++i, td++) {
      auto &x = td->first;  // test data
      auto &y = td->second; // expected result
      std::vector<ublas::vector<double>> delta_nabla_b(b0);
      std::vector<ublas::matrix<double>> delta_nabla_w(w0);
      backprop(x, y, delta_nabla_b, delta_nabla_w);
      for (auto k = 0; k < biases.size(); ++k) {
        nabla_b[k] += delta_nabla_b[k];
        nabla_w[k] += delta_nabla_w[k];
      }
    }
    for (auto i = 0; i < biases.size(); ++i) {
      biases[i] -= eta / mini_batch_size * nabla_b[i];
      weights[i] -= eta / mini_batch_size * nabla_w[i];
    }
  }
  // Populates the gradient for the cost function
  // for the biases in the vector nabla_b
  // and the weights in nabla_w
  void backprop(const ublas::vector<double> &x,
                const ublas::vector<double> &y,
                BiasesVector &nabla_b,
		WeightsVector &nabla_w) {
    auto activation = x;
    std::vector<ublas::vector<double>> activations; // Stores the activations of each layer
    activations.push_back(x);
    std::vector<ublas::vector<double>> zs; // The z vectors layer by layer
    for (auto i = 0; i < biases.size(); ++i) {
      ublas::vector<double> z = prod(weights[i], activation) + biases[i];
      zs.push_back(z);
      activation = z;
      sigmoid(activation);
      activations.push_back(activation);
    }
    // backward pass
    auto iActivations = activations.end() - 1;
    auto izs = zs.end() - 1;
    sigmoid_prime(*izs);
    ublas::vector<double> delta
      = element_prod(cost_derivative(*iActivations, y), *izs);
    auto ib = nabla_b.end() - 1;
    auto iw = nabla_w.end() - 1;
    *ib = delta;
    iActivations--;
    *iw = outer_prod(delta, trans(*iActivations));

    auto iWeights = weights.end();
    while (iActivations != activations.begin()) {
      izs--;
      iWeights--;
      iActivations--;
      ib--;
      iw--;
      sigmoid_prime(*izs);
      delta = element_prod(prod(trans(*iWeights), delta), *izs);
      *ib = delta;
      *iw = outer_prod(delta, trans(*iActivations));
    }
  }
  // Return the number of test inputs for which the neural
  //	network outputs the correct result. Note that the neural
  //	network's output is assumed to be the index of whichever
  //	neuron in the final layer has the highest activation.
  int evaluate(const std::vector<TrainingData> &td) const {
    return (int) count_if
      (td.begin(), td.end(),
       [this](const TrainingData &testElement) {
         auto res = feedforward(testElement.first);
         return (std::distance(res.begin(),
                               max_element(res.begin(),
                                           res.end()))
                 ==
                 std::distance(testElement.second.begin(),
                               max_element(testElement.second.begin(),
                                           testElement.second.end())));
                                          });
  }
  
  // Return the vector of partial derivatives \partial C_x /
  //	\partial a for the output activations.
  ublas::vector<double> cost_derivative
  (const ublas::vector<double> &output_activations,
   const ublas::vector<double> &y) const {
    return output_activations - y;
  }
};

#endif
