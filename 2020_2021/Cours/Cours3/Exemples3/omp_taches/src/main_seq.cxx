#include "arguments.hxx"
#include <iostream>

#define LEVEL 10

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
  Arguments A(argc, argv);
  int n = A.Get("n", 48);

  std::cout << "fib(" << n << ") = " 
            << fib_seq(n) << std::endl;
}
