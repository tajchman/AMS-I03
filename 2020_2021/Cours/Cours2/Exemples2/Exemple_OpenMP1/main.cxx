#include <iostream>

int main() {

  #pragma omp parallel
  {
    std::cout << "Bonjour" << std::endl;
  }
  
  return 0;
 }
