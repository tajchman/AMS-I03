#include <cstdlib>
#include <string>
#include <cstdio>
#include <cmath>
#include "calcul.h"

double * alloue (int n)
{
  double * v = new double [n*n];
  return v;
}

double * alloue_work(int n)
{
  return NULL;
}

void libere(double ** u)
{
  delete [] (*u);

  *u = NULL;
}

double difference (const double * u,
                   const double * v,
		   double *,
                   int n)
{
  int i, nn=n*n;
  double somme = 0.0;
  for (i=0; i<nn; i++)
      somme += fabs(u[i] - v[i]);
  
  return somme;
}
