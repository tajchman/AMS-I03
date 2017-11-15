#include <stdlib.h>
#include <stdio.h>
#include "util.h"

int main(int argc, char **argv)
{
  size_t i, N = memavail(0.5)/sizeof(int);

  printf("Ex 1 N = %ld\n\n", N);


  printf("Ex 1\n\n");

  printf("Reservation memoire\n");
  int *A = (int *) malloc(sizeof(int) * N);
  
  wait();

  printf("Initialisation\n");
  for(i=0; i<N; i++)
    A[i] = i*2;

  wait();
  free(A);
  
  return 0;
}
