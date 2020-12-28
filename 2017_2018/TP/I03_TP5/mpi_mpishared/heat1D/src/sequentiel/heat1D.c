#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void init(double *x, double *u, int n)
{
  int i;
  for (i = -1; i < n+1; ++i) {
    x[i] = (1.0 * (i+1))/(n+1);
    u[i] = x[i] > 0.3 && x[i] < 0.6 ? 1.0 : 0.0;
  }
}

double iteration(double *u, double *v, int n, double dt, double dx)
{
  double l = dt/(dx*dx);
  double d, diffmax = 0.0;
  int i;
  
  for (i = 0; i < n; ++i) {
    v[i]= u[i] + l*(u[i+1]-2*u[i]+u[i-1]);
    d = fabs(v[i]-u[i]);
    if (d > diffmax) diffmax = d;
  }
  v[-1] = u[-1];
  v[n] = u[n];

  return diffmax;
}

void print(double *x, double *u, int n, int step)
{
  char s[100];
  int i;
  FILE * f;

  sprintf(s, "u_%0d.dat", step);
  f = fopen(s, "w");;
  for (i = -1; i < n+1; ++i) {
    fprintf(f, "%12.6g %14.7g\n", x[i], u[i]);
  }
  fclose(f);	   
}

int main(int argc, char **argv) {

  int i, n = argc > 1 ? strtol(argv[1], NULL, 10) : 100000;
  int step, maxsteps = argc > 2 ? strtol(argv[2], NULL, 10) : 10000;
  int dstep;
  double dx = 1.0/(n-1);
  double dt = 0.4*dx*dx;
  double debug = 0;
  
  double *_x = (double *) malloc(sizeof(double) * (n+2));
  double *_u = (double *) malloc(sizeof(double) * (n+2));
  double *_v = (double *) malloc(sizeof(double) * (n+2));

  double * x = _x+1;
  double * u = _u+1;
  double * v = _v+1;
  double diff;
  
  init(x, u, n);

  dstep = maxsteps/100;
  if (dstep == 0) dstep = 1;
  
  if (debug) print(x, u, n, 0);
  
  for (step = 0; step < maxsteps; ++step)
  {
    diff = iteration(u, v, n, dt, dx);

    if (debug && ((step+1) % dstep) == 0) print(x, v, n, step+1);
    fprintf(stderr, "step %5d : diff = %12.5g\n", step, diff);
    
    double * temp = v;
    v = u;
    u = temp;
  }
  
  fprintf(stderr, "\n\n");
  if (debug) print(x, v, n, step+1);
  free(_x);
  free(_u);
  free(_v);
  
  return 0;
}
