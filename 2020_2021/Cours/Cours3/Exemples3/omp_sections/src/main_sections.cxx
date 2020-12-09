#include "arguments.hxx"
#include "timer.hxx"
#include "pause.hxx"
#include <iostream>
#include <iomanip>
#include <omp.h>

int level_max;

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

long fib_sections(int n, int lv)
{

/*
#pragma omp critical
  std::cerr << std::setw(lv *2 + 1) << " "
            << " on thread " << omp_get_thread_num()
            << " n = " << n 
            << std::endl;
*/

  if (lv > level_max) return fib_seq(n);
    
  long i, j;
  if (n<2)
    return n;
  else {
    lv++;
    
    omp_set_num_threads(2);
    #pragma omp parallel sections shared(i,j,n, lv)
    {
      #pragma omp section
      i = fib_sections(n-1, lv);		
      #pragma omp section
      j = fib_sections(n-2, lv);		
    }
    return i + j;
  }
}

int main(int argc, char **argv)
{
  Timer T;
  T.start();

  Arguments A(argc, argv);
  int n = A.Get("n", 48);
  level_max = A.Get("levels", 4);

  omp_set_nested(1);

  long f = fib_sections(n, 0/*, 0*/);
  std::cout << "fib(" << n << ") = " << f << std::endl;

  T.stop();
  std::cout << "CPU time : " << T.elapsed() << "s" 
            << std::endl;
}
