#include "random.h"
#include <stdlib.h>

static int a;
static double b;
static unsigned int * seeds;

void initRandom(int seed, int min, int max, int nThreads)
{
  int i;
  seeds = (unsigned int *) malloc(sizeof(unsigned int) * nThreads);
  for (i=0; i<nThreads; i++)
    seeds[i] = seed + i;
  b = ((double)(max-min))/RAND_MAX;
  a = min;
}

int nextRnd(int * seed)
{
  return a + (int)(rand_r(seed) * b);
}

int getSeed(int iThread) {
  return seeds[iThread];
}

void cleanRandom()
{
  free(seeds);
}
