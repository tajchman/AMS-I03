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

double * alloue_work(int n)
{
  return NULL;
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

void laplacien(double * v, const double * u, double dx, int n)
{
  int i,j;
  double L = 1.0/(dx*dx);

  for (i=1; i<n-1; i++)
    for (j=1; j<n-1; j++) {
      v[i*n + j] = - L * (4*u[i*n + j]
		    - u[(i+1)*n + j] - u[(i-1)*n + j]
		    - u[i*n + (j+1)] - u[i*n + (j-1)]);
    }
}

void calcul_forces(double * forces,
                   const double * u,
                   int n) {

  int i, nn = n*n;
  for (i=0; i<nn; i++) {
  
    uu = u[i];

    if (i==3*n/4 && j>n/4 && j<3*n/4)
      f = (0.25 - uu*uu);
    else if (i==n/4 && j>n/4 && j<3*n/4)
      f = -(0.25 - uu*uu);
    else
      f = 0.0;

    forces[i] = f;
  }
}


void variation    (double * u_next,
                   const double * u_current,
                   const double * u_diffuse,
                   const double * forces,
                   double dt, int n) {
  int i, nn = n*n;
  
  for (i=0; i<nn; i++)
    u_next[i] = u_current[i] + dt*(u_diffuse[i] + forces[i]);
}


double difference (const double * u,
                   const double * v,
                   int n)
{
  int i, nn=n*n;
  double somme = 0.0;
  for (i=0; i<nn; i++)
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
