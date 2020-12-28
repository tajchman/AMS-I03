#include "arguments.hxx"
#include "timer.hxx"
#include "add_tbb.hxx"
#include "tbb/parallel_for.h"
#include "tbb/global_control.h"
#include "tbb/task_scheduler_init.h"
#include <iostream>

int main(int argc, char **argv)
{
  Arguments A(argc, argv);

  int n = A.Get("n", 100000000);
  int nThreads = A.Get(
    "threads", 
    tbb::task_scheduler_init::default_num_threads());
  int grain = A.Get("grain", 1);

  tbb::global_control Control
    (tbb::global_control::max_allowed_parallelism, nThreads);

  Timer T;
  T.start();

  std::vector<double> a(n, 1.0), b(n, 2.0);
  AddTBB Add(a, b, a);

  int nIt = 100;
  for (int it=0; it<nIt; it++)
    tbb::parallel_for
      (tbb::blocked_range<int>(0, n, grain), 
       Add);

  T.stop();
   
  std::cout << "Threads : " << nThreads
            << " Grain : " << grain << std::endl
            << "CPU time : " 
            << T.elapsed() << " s" 
            << " erreur : " << a[n/2] - (1 + 2*nIt)
            << std::endl;
}
