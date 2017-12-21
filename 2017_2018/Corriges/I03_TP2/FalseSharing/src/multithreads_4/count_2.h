#ifndef NVAL
#define NVAL 20
#endif

double * countAllocate  (int padding);
void     countNormalize (double *s,int padding);
void     countPrint     (double *s,int padding);
void     countDelete    (double **s);
