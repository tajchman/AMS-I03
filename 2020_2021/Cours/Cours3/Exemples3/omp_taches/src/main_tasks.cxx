#include "arguments.hxx"
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

#define LEVEL 10
//int ntasks;

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

long fib_tasks(int n, int lv)
{
  if (lv > LEVEL) return fib_seq(n);
  
// #pragma omp critical
//  std::cerr << "tasks : " << ++ntasks << std::endl;
  
  long i, j;
  if (n<2)
    return n;
  else
    {
      lv++;
#pragma omp task shared(i, n, lv)
      i=fib_tasks(n-1, lv);
#pragma omp task shared(j, n, lv)
      j=fib_tasks(n-2, lv);

#pragma omp taskwait

       return i+j;
    }
}
int main(int argc, char **argv)
{
  Arguments A(argc, argv);
  int n = A.Get("n", 48);

#ifdef _OPENMP
  int nthreads = A.Get("threads", 4);
  omp_set_num_threads(nthreads);
#endif

#pragma omp parallel
  {
#pragma omp single
    {
//    ntasks = 0;
    std::cout << "fib(" << n << ") = " 
              << fib_tasks(n, 0) << std::endl;
    }
  }
}
