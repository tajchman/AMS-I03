#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sum(const double *a, size_t n)
{
    if (n == 0) {
        return 0;
    }
    else if (n == 1) {
        return *a;
    }

    // recursive case
    size_t half = n / 2;
    return sum(a, half) + sum(a + half, n - half);
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
