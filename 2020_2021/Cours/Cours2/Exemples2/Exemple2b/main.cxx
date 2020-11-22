#include <iostream>
#include <omp.h>

int main() {

  #pragma omp parallel
  {
    std::cout << "Bonjour, je suis le thread " << omp_get_thread_num() << std::endl;
  }
  
  return 0;
 }
