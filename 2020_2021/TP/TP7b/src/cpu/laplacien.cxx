#include <cmath>
#include "calcul.h"

void laplacien(double * v, const double * u,
	       double dx, int n)
{
  int i,j;
  double L = 0.5/(dx*dx);

  for (i=1; i<n-1; i++)
    for (j=1; j<n-1; j++) {
      v[i*n + j] = -L * (4*u[i*n + j]
			  - u[(i+1)*n + j] - u[(i-1)*n + j]
			  - u[i*n + (j+1)] - u[i*n + (j-1)]);
    }
}

