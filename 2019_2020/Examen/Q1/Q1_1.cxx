#include <vector>
#include <iostream>
#include "timer.hxx"
#include "matmult1.hxx"

int main()
{
  int N=40;

  std::vector<double> A(N*N, 1.0), V(N, 0.5), W(N, 0.0);

  Timer T;

  T.start();
  
  matmult1(A, V, W);

  T.stop();
  std::cerr << T.elapsed() << std::endl;

  return 0;
}
