#include <cstdlib>
#include <string>
#include <cstdio>
#include <cmath>
#include "calcul.h"

double * alloue (int n)
{
  double * v = new double [n*n];
  return v;
}

void libere(double ** u)
{
  delete [] (*u);

  *u = NULL;
}

double * init(int n)
{
  double * u = alloue(n);

  int i;
  for (i=0; i<n*n; i++)
      u[i] = 0.0;

  return u;
}

void   iteration(double * v, const double * u, double dt, int n)
{
  int i,j;
  double lambda = 0.25, f;
  
  for (i=1; i<n-1; i++)
    for (j=1; j<n-1; j++) {
      v[i*n + j] = u[i*n + j]
	- lambda * (4*u[i*n + j]
		    - u[(i+1)*n + j] - u[(i-1)*n + j]
		    - u[i*n + (j+1)] - u[i*n + (j-1)]);
      double uu = u[i*n + j];
      if (i==3*n/4 && j>n/4 && j<3*n/4)
	f = (0.25 - uu*uu);
      else if (i==n/4 && j>n/4 && j<3*n/4)
	f = -(0.25 - uu*uu);
      else
	f = 0.0;

      printf("i = %03d j = %03d f = %16.10g\n", i, j, f);
      v[i*n + j] += f*dt;
    }
}

double difference(const double * u, const double * v, int n)
{
  int i;
  double somme = 0.0;
  for (i=0; i<n*n; i++)
      somme += fabs(u[i] - v[i]);
  
  return somme;
}

void save(const char *name, const double *u, int n)
{
  int i,j;

  std::string s = "cpu_";
  s += name;
  
  FILE * f = fopen(s.c_str(), "w");
  
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++)
      fprintf(f, "%g %g %g\n", i*1.0/n, j*1.0/n, u[i*n+j]);
    fprintf(f,"\n");
  }
  fprintf(f,"\n");

  fclose(f);
}
