#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

void init(double *x, double *u, int n, int i1, int i2)
{
  int i;
  for (i = i1-1; i < i2+1; ++i) {
    x[i-i1] = (1.0 * (i+1))/(n+1);
    u[i-i1] = x[i-i1] > 0.3 && x[i-i1] < 0.6 ? 1.0 : 0.0;
  }
}

double iteration(double *u, double *v, int n, double dt, double dx)
{
  double l = dt/(dx*dx);
  double d, diffmax = 0.0, diff_global;
  int i;
  
  for (i = 0; i < n; ++i) {
    v[i]= u[i] + l*(u[i+1]-2*u[i]+u[i-1]);
    d = fabs(v[i]-u[i]);
    if (d > diffmax) diffmax = d;
  }
  v[-1] = u[-1];
  v[n] = u[n];

  MPI_Allreduce(&diffmax, &diff_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  diffmax = diff_global;
  return diffmax;
}

void echanges(double *v, int n, int rank, int size)
{
  MPI_Status status;
  MPI_Barrier(MPI_COMM_WORLD);
  
  if (rank > 0) {
    MPI_Sendrecv(&v[0],   1, MPI_DOUBLE, rank-1, 0,
		 &v[-1],  1, MPI_DOUBLE, rank-1,   0,
		 MPI_COMM_WORLD, &status);
  }
  if (rank < size-1) {
    MPI_Sendrecv(&v[n-1], 1, MPI_DOUBLE, rank+1, 0,
		 &v[n],   1, MPI_DOUBLE, rank+1, 0,
		 MPI_COMM_WORLD, &status);
  }
}

void print(double *x, double *u, int n, int step)
{
  char s[100];
  int i;
  FILE * f;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  sprintf(s, "u_mpi_%0d_%0d.dat", step, rank);
  f = fopen(s, "w");;
  for (i = -1; i < n+1; ++i)
    fprintf(f, "%12.6g %14.7g\n", x[i], u[i]);
  fclose(f);	   
}

int main(int argc, char **argv) {

  int i, n = argc > 1 ? strtol(argv[1], NULL, 10) : 100000;
  int n_local, i1, i2;
  int step, maxsteps = argc > 2 ? strtol(argv[2], NULL, 10) : 10000;
  int dstep;
  double dx = 1.0/(n-1);
  double dt = 0.4*dx*dx;
  double debug = 0;

  int rank, size;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  i1 = rank * (n/size);
  i2 = i1 + (n/size);
  if (rank == size-1) i2 = n;
  n_local = i2 - i1;
  
  double *_x = (double *) malloc(sizeof(double) * (n_local+2));
  double *_u = (double *) malloc(sizeof(double) * (n_local+2));
  double *_v = (double *) malloc(sizeof(double) * (n_local+2));

  double * x = _x+1;
  double * u = _u+1;
  double * v = _v+1;
  double diff;
  
  init(x, u, n, i1, i2);

  dstep = maxsteps/100;
  if (dstep == 0) dstep = 1;
  
  if (debug) print(x, u, n_local, 0);
  
  for (step = 0; step < maxsteps; ++step)
  {
    diff = iteration(u, v, n_local, dt, dx);
    
    echanges(v, n_local, rank, size);

    if (debug && ((step+1) % dstep) == 0)
      print(x, v, n_local, step+1);
    if (rank == 0)
      fprintf(stderr, "step %5d : diff = %12.5g\n", step, diff);
    
    double * temp = v;
    v = u;
    u = temp;
  }
  
  if (rank == 0)
    fprintf(stderr, "\n\n");
  print(x, v, n_local, step+1);
  free(_x);
  free(_u);
  free(_v);

  MPI_Finalize();
  
  return 0;
}
