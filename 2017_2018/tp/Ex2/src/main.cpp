#include <memory>
#include <iostream>
#include "util.h"

int main(int argc, char **argv)
{
  size_t i, N = memavail(0.5)/sizeof(double);

  std::cout << "Ex 2 N = " << N << std::endl << std::endl;

  std::cout <<"Reservation memoire" << std::endl;
  double *A = new double[N];

  wait();

  std::cout << "Initialisation" << std::endl;
  for(i=0; i<N; i++)
    A[i] = i*2.0;
  
  wait();
  
  delete [] A;
  
  return 0;
}
