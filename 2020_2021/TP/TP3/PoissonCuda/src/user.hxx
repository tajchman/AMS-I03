#ifndef __USER_HXX
#define __USER_HXX

#include "values.hxx"

void iterationCuda(
    Values & v, const Values & u,
    int imin, int imax, 
    int jmin, int jmax,
    int kmin, int kmax);

#endif
