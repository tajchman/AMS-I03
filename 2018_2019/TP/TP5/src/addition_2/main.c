#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "add.h"

void addition_CPU(double *w, double *u, double *v, int n)
{
  int i;
  for (i=0; i<n; i++)
    w[i] = u[i] + v[i];
}

int main(int argc, char **argv)
{
  int n = 10000000, i;
  size_t bytes = n*sizeof(double);
  double somme, diff;
  
  double *u = (double *) malloc(bytes);
  double *v = (double *) malloc(bytes);
  double *w1 = (double *) malloc(bytes);
  double *w2 = (double *) malloc(bytes);
  
  for( i = 0; i < n; i++ ) {
    u[i] = sin(i)*sin(i);
    v[i] = cos(i)*cos(i);
  }
  
  addition_CPU(w1, u, v, n);
  addition_GPU(w2, u, v, n);
  
  somme = 0;
  diff = 0;
  for(i=0; i<n; i++) {
    somme += w1[i];
    diff += fabs(w1[i] - w2[i]);
  }
  printf("resultat : %g erreur : %g\n", somme/n, diff);
  
  free(u);
  free(v);
  free(w2);
  free(w1);
  
  return 0;
}
