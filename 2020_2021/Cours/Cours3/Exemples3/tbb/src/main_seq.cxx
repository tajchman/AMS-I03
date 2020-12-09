#include "arguments.hxx"
#include "timer.hxx"
#include "add_seq.hxx"
#include <iostream>

int main(int argc, char **argv)
{
  Arguments A(argc, argv);
  int n = A.Get("n", 10000000);

  Timer T;
  T.start();

  std::vector<double> a(n, 1.0), b(n, 2.0);
  Additionneur Add(a, b, a);

  int nIt = 200;
  for (int it=0; it<nIt; it++)
    Add(); 

  T.stop();
   
  std::cout << "CPU time : " 
            << T.elapsed() << " s" 
            << " erreur : " << a[n/2] - (1 + 2*nIt)
            << std::endl;
}
