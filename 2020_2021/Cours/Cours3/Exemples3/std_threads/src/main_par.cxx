#include "arguments.hxx"
#include "timer.hxx"
#include "add_par.hxx"
#include <iostream>
#include <thread>

int main(int argc, char **argv)
{
  Arguments A(argc, argv);

  int n = A.Get("n", 100000000);
  int nThreads = A.Get("threads", 1);

  Timer T;
  T.start();

  std::vector<double> a(n, 1.0), b(n, 2.0);
  AddPartial Add(a, b, a);

  int nIt = 200;
  int iThread;
  int dn = n/nThreads;

  for (int it=0; it<nIt; it++) {
    std::vector<std::thread> threads;
    int i0, i1 = 0;
    for (iThread = 0; iThread<nThreads-1; iThread++)
    {
      i0 = i1;
      i1 = i0 + dn;
      threads.push_back(std::thread(Add, i0, i1));
    }

    Add(i1, n);

    for (auto & th : threads)
      th.join();

  }  
  T.stop();
   
  std::cout << "Threads : " << nThreads
            << " CPU time : " 
            << T.elapsed() << " s" 
            << " erreur : " << a[n/2] - (1 + 2*nIt)
            << std::endl;
}
