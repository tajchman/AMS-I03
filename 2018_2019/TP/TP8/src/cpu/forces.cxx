#include "calcul.h"
#include <cmath>


void calcul_forces(double * f,
                   const double * u,
                   int n) {

  int i, j;
  double uu, ff;

  int
    i0 = n/10,
    i1 = n-n/10,
    j0 = n/2 - n/10,
    j1 = n/2 + n/10;
  
  for (i=1; i<n-1; i++)
    for (j=1; j<n-1; j++) {
      
      uu = u[i*n+j];
      
      if (i>i0 && i<i1 && j>j0 && j<j1)
	ff = (20 - 20*uu*uu);
      else
	ff = 0.0;

      f[i*n + j] = ff;
  }
}


