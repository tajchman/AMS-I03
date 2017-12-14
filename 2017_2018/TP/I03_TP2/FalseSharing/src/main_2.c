#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>

#if defined(_OPENMP)
#include <omp.h>
#endif
#define NVAL 20

void setOptions(int argc, char **argv, int * nThreads, long long * nSamples)
{
  char c;
  *nSamples = 1000000;
  *nThreads = 1;
  while ((c = getopt(argc , argv, "n:t:")) != -1)
    switch (c) {
    case 'n':
      *nSamples = 1000L * strtoll(optarg, NULL, 10);
      break;

    case 't':
#if defined(_OPENMP)
      *nThreads = strtol(optarg, NULL, 10);
#endif
      break;

    default:
      abort ();
    }
}

void initGenere()
{
#pragma omp parallel private(i) shared(s)
  _Thread_local unsigned int seed = time(NULL);
  srand(time(NULL));
}

int genere()
{
  int r = rand();
  r = (int)(((double)r)/RAND_MAX * NVAL);
  return r;
}

int main(int argc, char **argv)
{
  fprintf(stderr, "TP2 %s\n", argv[0]);
  long long nSamples;
  int nThreads;
  double *s, sum;
  int i;
  long iSamples;

  s = (double *) malloc(sizeof(double) * NVAL);
  setOptions(argc, argv, &nThreads, &nSamples);
  
#if defined(_OPENMP)
  omp_set_num_threads(nThreads);
#endif
  
  printf(" %2d threads\n", nThreads);
  printf(" %ld samples\n", nSamples);
  
  for (i=0; i<NVAL; i++)
    s[i] = 0.0;

  double ** u = (double **) malloc(sizeof(double *) * nThreads);
                                   
#pragma omp parallel private(i) shared(s)
  {
#if defined(_OPENMP)
    int iThread = omp_get_num_threads();
#else
    int iThread = 1;
#endif
    double *t = (double *) malloc(sizeof(double) * NVAL);
    
    u[iThread] = t;
    for (i=0; i<NVAL; i++)
      t[i] = 0;   
#pragma omp for 
    for (iSamples=0; iSamples<nSamples; iSamples++) {
      i = genere();
      t[i] += 1.0;
    }
#pragma omp critical
    for (i=0; i<NVAL; i++)
      s[i] += t[i];   
  }
  
  sum = 0.0;
  for (i=0; i<NVAL; i++)
    sum += s[i];
  for (i=0; i<NVAL; i++)
    s[i] /= sum;

  printf("s : \n"); 
  for (i=0; i<NVAL; i++)
    printf("%12.6f", i, s[i]);
  printf("\n");
  
  free(s);
  return 0;
}
