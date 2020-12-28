#include "count_2.h"
#include <stdio.h>
#include <stdlib.h>

double * countAllocate(int padding) {

  double * s = (double *) malloc(sizeof(double) * NVAL * padding);

  for (int i=0; i<NVAL; i++)
    s[i*padding] = 0.0;

  return s;
}

void countNormalize(double *s, int padding) {

  double sum = 0.0;
  int i;
  
  for (i=0; i<NVAL; i++)
    sum += s[i*padding];
  for (i=0; i<NVAL; i++)
    s[i*padding] /= sum;
}

void countPrint(double *s, int padding) {

  int i;
  
  printf("s : "); 
  for (int i=0; i<NVAL; i++) {
    if (i % 5 == 0) printf("\n");
    printf("%12.6f", s[i*padding]);
  }
  printf("\n");
}

void countDelete(double **s)
{
  free(*s);
  *s = NULL;
}
