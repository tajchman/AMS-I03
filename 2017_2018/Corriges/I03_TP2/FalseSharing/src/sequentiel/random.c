#include <stdlib.h>

static int a;
static double b;

void initRandom(int seed, int min, int max)
{
  srand(seed);
  b = ((double)(max-min))/RAND_MAX;
  a = min;
}

int nextRnd()
{
  return a + (int)(rand() * b);
}
