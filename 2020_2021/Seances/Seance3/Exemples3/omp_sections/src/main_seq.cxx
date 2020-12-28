#include "arguments.hxx"
#include "timer.hxx"
#include <iostream>

long fib_seq(int n)
{
  long i, j;
  if (n<2)
    return n;
  else
    {
       i=fib_seq(n-1);
       j=fib_seq(n-2);

       return i+j;
    }
}

int main(int argc, char **argv)
{
  Timer T;
  T.start();

  Arguments A(argc, argv);
  int n = A.Get("n", 48);

  std::cout << "fib(" << n << ") = " 
            << fib_seq(n) << std::endl;

  T.stop();
  std::cout << "CPU time : " << T.elapsed() << "s" 
            << std::endl;
}
