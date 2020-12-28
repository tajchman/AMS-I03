#include "count.h"
#include <stdio.h>
#include <stdlib.h>

double * countAllocate() {

  double * s = (double *) malloc(sizeof(double) * NVAL);

  for (int i=0; i<NVAL; i++)
    s[i] = 0.0;

  return s;
}

void countNormalize(double *s) {

  double sum = 0.0;
  int i;
  
  for (i=0; i<NVAL; i++)
    sum += s[i];
  for (i=0; i<NVAL; i++)
    s[i] /= sum;
}

void countPrint(double *s) {

  int i;
  
  printf("s : "); 
  for (int i=0; i<NVAL; i++) {
    if (i % 5 == 0) printf("\n");
    printf("%12.6f", s[i]);
  }
  printf("\n");
}

void countDelete(double **s)
{
  free(*s);
  *s = NULL;
}
