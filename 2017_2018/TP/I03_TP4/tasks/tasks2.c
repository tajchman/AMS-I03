#include <stddef.h>

#define CUTOFF 100

double parallel_sum(const double *, size_t);
double serial_sum(const double *, size_t);

double sum(const double *a, size_t n)
{
    double r;

    #pragma omp parallel
    #pragma omp single nowait
    r = parallel_sum(a, n);
    return r;
}

double parallel_sum(const double *a, size_t n)
{
    // cas trop petit pour etre calcule en parallele
    if (n <= CUTOFF) {
        return serial_sum(a, n);
    }

    // cas recursif
    double x, y;
    size_t half = n / 2;

    #pragma omp task shared(x)
    x = parallel_sum(a, half);
    #pragma omp task shared(y)
    y = parallel_sum(a + half, n - half);
    #pragma omp taskwait
    x += y;

    return x;
}

double serial_sum(const double *a, size_t n)
{
    if (n == 0) {
        return 0.;
    }
    else if (n == 1) {
        return a[0];
    }

    size_t half = n / 2;
    return serial_sum(a, half) + serial_sum(a + half, n - half);
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
