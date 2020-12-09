#include "arguments.hxx"
#include <iostream>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif

int level_max;
// int ntasks;

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

long fib_tasks(int n, int lv/*, int task_pere*/)
{
//  ntasks++;
//  int task_id = ntasks;

/*
  #pragma omp critical
  std::cerr << std::setw(lv *2 + 1) << " "
            << "task : " << task_id
            << " (from " << task_pere << ")" 
            << " on thread " << omp_get_thread_num()
            << " n = " << n 
            << std::endl;
*/
  if (lv >= level_max) return fib_seq(n);
    
  long i, j;
  if (n<2)
    return n;
  else
    {
      lv++;
#pragma omp task shared(i, n, lv)
        i=fib_tasks(n-1, lv/*, task_id*/);

#pragma omp task shared(j, n, lv)
      j=fib_tasks(n-2, lv/*, task_id*/);

#pragma omp taskwait

       return i+j;
    }
}

int main(int argc, char **argv)
{
  Arguments A(argc, argv);
  int n = A.Get("n", 48);
  level_max = A.Get("levels", 4);

#ifdef _OPENMP
  int nthreads = A.Get("threads", 4);
  omp_set_num_threads(nthreads);
#endif

#pragma omp parallel
  {
#pragma omp single
    {
//    ntasks = 0;
    long f = fib_tasks(n, 0/*, 0*/);
    std::cout << "fib(" << n << ") = " << f << std::endl;
    }
  }
}
