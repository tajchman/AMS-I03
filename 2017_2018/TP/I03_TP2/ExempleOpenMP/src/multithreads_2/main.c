#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "count.h"
#include "random.h"
#include "parameters.h"

int main(int argc, char **argv)
{
  long long nSamples;
  int nThreads = 1;
  double *s;
  long iSamples;
  void * params;
  
#if defined(_OPENMP)
#pragma omp parallel
#pragma omp master
  nThreads = omp_get_num_threads();
#endif
  
  params = parseArgs(argc, argv);
  printf("\nTP2 %s\n", argv[0]);

  nSamples = getLong(params, "n", 1000L * 200000L);
  printf("\n %lld samples\n\n", nSamples);
  nThreads = getInt(params, "threads", nThreads);
  printf(" %d threads\n\n", nThreads);

#if defined(_OPENMP)
  omp_set_num_threads(nThreads);
#endif
  
  s = countAllocate();
  initRandom(time(NULL), 0, NVAL, nThreads);
  
#pragma omp parallel private(iSamples) default(shared)
  {
    int iThread = omp_get_thread_num();
    int seed = getSeed(iThread);
    double * s_local = countAllocate();
    
#pragma omp for
    for (iSamples=0; iSamples<nSamples; iSamples++) {
      s_local[nextRnd(&seed)] += 1.0;
    }

#pragma omp critical
    {
      for (int i=0; i<NVAL; i++)
	s[i] += s_local[i];
    }
    
    countDelete(&s_local);
  }
  countNormalize(s);
  countPrint(s);
  
  countDelete(&s);
  cleanRandom();
  
  return 0;
}
