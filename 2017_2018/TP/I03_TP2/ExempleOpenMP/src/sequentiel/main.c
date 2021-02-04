#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>

#include "count.h"
#include "random.h"
#include "parameters.h"

int main(int argc, char **argv)
{
  long long nSamples;
  double *s;
  long iSamples;
  void * params;
  
  
  params = parseArgs(argc, argv);
  printf("\nTP2 %s\n", argv[0]);
  
  nSamples = getLong(params, "n", 1000L * 200000L);
  printf("\n%lld samples\n\n", nSamples);
    
  s = countAllocate();
  initRandom(time(NULL), 0, NVAL);
  
  for (iSamples=0; iSamples<nSamples; iSamples++) {
    s[nextRnd()] += 1.0;
  }

  countNormalize(s);
  countPrint(s);
  
  countDelete(&s);
  
  return 0;
}
