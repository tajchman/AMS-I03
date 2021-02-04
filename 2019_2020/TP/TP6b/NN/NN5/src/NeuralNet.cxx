// NeuralNet1.cpp : Console application to demonstrate Machine learning
//
// An example written to implement the stochastic gradient descent learning algorithm
// for a feedforward neural network. Gradients are calculated using backpropagation.
//
// Code is written to be a C++ version of network.py from http://neuralnetworksanddeeplearning.com/chap1.html
// Variable and functions names follow the names used in the original Python
//
// Uses the boost ublas library for linear algebra operations

#include "timer.hxx"
#include "mnist_loader.h"
#include <cmath>
#include <iostream>
#include "Network.h"

#ifdef _WIN32
   constexpr char sep = '\\';
#else
   constexpr char sep = '/';   
#endif

std::string getPathName(const std::string& s) {

   size_t i = s.rfind(sep, s.length());
   if (i != std::string::npos) {
      return(s.substr(0, i));
   }

   return("./");
}

int main(int argc, char **argv) {

  
  std::string DataDir = getPathName(getPathName(getPathName(argv[0])));
  DataDir += sep;
  DataDir += "Data";
  DataDir += sep;
  
  std::vector<Network::TrainingDataCPU> trainData, testData;

  Timer t, t_total;
  
  try {
  // Load training data
    t.start();
    mnist_loader
      (DataDir + "train-images.idx3-ubyte",
       DataDir + "train-labels.idx1-ubyte",
       trainData);
    t.stop();
    std::cout << "Load training data in " << t.elapsed() << " s" << std::endl;

    // Load test data

    t.reinit();
    t.start();
    mnist_loader
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
  net.SGD(trainData, 10, 50, 3.0, testData);

  t_total.stop();
  std::cout << "\nTotal: " << t_total.elapsed() << " s\n" << std::endl;

  return 0;
}
