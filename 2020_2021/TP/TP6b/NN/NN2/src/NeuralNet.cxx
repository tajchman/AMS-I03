// NeuralNet1.cpp : Console application to demonstrate Machine learning
//
// An example written to implement the stochastic gradient descent learning algorithm
// for a feedforward neural network. Gradients are calculated using backpropagation.
//
// Code is written to be a C++ version of network.py from http://neuralnetworksanddeeplearning.com/chap1.html
// Variable and functions names follow the names used in the original Python
//
// Uses the boost ublas library for linear algebra operations

#include <cmath>
#include <iostream>
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "GetPot.hxx"
#include "OS.hxx"
#include "timer.hxx"
#include "mnist_loader.h"
#include "Network.hxx"

void usage(const char *s)
{
  std::cerr << "usage " << s << " [batch=<int>] [epochs=<int>]" << std::endl;
  std::cerr << "   batch  : mini-batch size" << std::endl;
  std::cerr << "   epochs : number of epochs" << std::endl;
  std::exit(-1);
}

int main(int argc, char **argv) {

  GetPot opt(argc, argv);
  if (opt.options_contain("h") or opt.long_options_contain("help"))
    usage(argv[0]);
  
  int batch_size = opt("batch", 50);
  int epochs = opt("epochs", 10);
    
  Timer t_total;
  t_total.start();

  std::string DataDir = GetDirName(argv[0], 3);
  DataDir += sep;
  DataDir += "Data";
  DataDir += sep;
  
  std::vector<Network::TrainingData> trainData, testData;

  Timer t;
  
  try {
  // Load training data
    t.start();
    mnist_loader<double>
      (DataDir + "train-images.idx3-ubyte",
       DataDir + "train-labels.idx1-ubyte",
       trainData);
    t.stop();
    std::cout << "Load training data in " << t.elapsed() << " s" << std::endl;

    // Load test data

    t.reinit();
    t.start();
    mnist_loader<double>
      (DataDir + "t10k-images.idx3-ubyte",
       DataDir + "t10k-labels.idx1-ubyte",
       testData);
    t.stop();
    std::cout << "Load testing data in  " << t.elapsed() << " s" << std::endl;
  }
  catch (const char *ex) {
    std::cout << "Error: " << ex << std::endl;
    return 0;
  }
  
  Network net({ 784, 30, 10 });
  net.SGD(trainData, epochs, batch_size, 3.0, testData);

  t_total.stop();
  std::cout << "\nTotal: " << t_total.elapsed() << " s\n" << std::endl;

  return 0;
}
