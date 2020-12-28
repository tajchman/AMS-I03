#include <string>
#include "calcul.h"

double * init(int n)
{
  double * u = alloue(n);

  int i, j;

  for (i=0; i<n; i++)
    for (j=0; j<n; j++)
      if (j > 0.2*n && i > 0.4*n && i < 0.6*n && j < 0.8*n)
	u[i*n+j] = 1.0;
      else
	u[i*n+j] = 0.0;

  return u;
}

double * zero(int n)
{
  double * u = alloue(n);

  int i, j;

  for (i=0; i<n; i++)
    for (j=0; j<n; j++)
	u[i*n+j] = 0.0;

  return u;
}

