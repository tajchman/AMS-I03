#include "values.hxx"

double variationWrapper(Values & u, Values & v, 
                        double *& d_partialSum, int n);

void freeVariationData(double *& d_partialSum);