#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>

#include "count.h"
#include "random.h"

void setOptions(int argc, char **argv, long long * nSamples)
{
  char c;
  *nSamples = 1000L * 200000L;
  while ((c = getopt(argc , argv, "n:")) != -1)
    switch (c) {
    case 'n':
      *nSamples = 1000L * strtoll(optarg, NULL, 10);
      break;

    default:
      abort ();
    }
}

int main(int argc, char **argv)
{
  long long nSamples;
  double *s;
  long iSamples;
  
  fprintf(stderr, "\nTP2 %s\n", argv[0]);
  setOptions(argc, argv, &nSamples);

  printf(" %lld samples\n\n", nSamples);
    
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
