#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sum(const double *a, size_t n)
{
    // base cases
    if (n == 0) {
        return 0;
    }
    else if (n == 1) {
        return 1;
    }

    // cas recursif
    size_t half = n / 2;
    double x, y;

    #pragma omp parallel
    #pragma omp single nowait
    {
        #pragma omp task shared(x)
        x = sum(a, half);
        #pragma omp task shared(y)
        y = sum(a + half, n - half);
        #pragma omp taskwait
        x += y;
    }
    return x;
}


int main(int argc, char **argv)
{
  size_t n, i;
  double * x, s;


  n = argc > 1 ? strtol(argv[1], NULL, 10) : 10000000L;
  x = (double *) malloc(sizeof(double) * n);
  
  for (i=0; i<n; i++)
    x[i] = sin((M_PI * i) / n);

  s = sum(x, n)/n;
  fprintf(stderr, "somme = %g\n", s);

  return 1;
}
