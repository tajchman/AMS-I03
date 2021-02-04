#include "calcul.h"

void variation    (double * u_next,
                   const double * u_current,
                   const double * u_diffuse,
                   const double * forces,
                   double dt, int n) {
  int i, nn = n*n;
  
  for (i=0; i<nn; i++)
    u_next[i] = u_current[i] + dt*(u_diffuse[i] + forces[i]);
}
