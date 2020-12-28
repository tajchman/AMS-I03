/*
 * scheme.cxx
 *
 *  Created on: 5 janv. 2016
 *      Author: tajchman
 */
#include "scheme.hxx"
#include <iostream>
#include <cmath>
#include <omp.h>

inline double iterate_bloc(int i0, int i1, int j0, int j1, int k0, int k1,
			   int di, int dj, int dk,
			   const Values & u1, Values & u2,
			   double lambda, double mu)
{
  double d = 0.0;
  int i,j,k;

  for (i=i0; i<i1; i++)
    for (j=j0; j<j1; j++)
      for (k=k0; k<k1; k++) {
	  u2(i,j,k) = u1(i,j,k)
	  - lambda * (6*u1(i,j,k)
	  - u1(i+di,j,k) - u1(i-di,j,k)
	  - u1(i,j+dj,k) - u1(i,j-dj,k)
	  - u1(i,j,k+dk) - u1(i,j,k-dk))
	  - mu*(u1(i,j,k) - u1(i-di, j, k));
	  d += std::abs(u2(i,j,k) - u1(i,j,k));
      }
  return d;
}

double iterate(const Values & u1, Values & u2,
               double dt, Parameters &P)
{
  double du_sum = 0.0, du;
  double mu = 0.5*dt/P.dx(0);
  double dx2 = P.dx(0)*P.dx(0) + P.dx(1)*P.dx(1) + P.dx(2)*P.dx(2);
  double lambda = 0.25*dt/dx2;

  if (not P.diffusion()) lambda = 0.0;
  if (not P.convection())  mu = 0.0;
  
  int i, j, k;
  int   di = P.di(0),     dj = P.di(1),     dk = P.di(2);
  int imin = P.imin(0), jmin = P.imin(1), kmin = P.imin(2);
  int imax = P.imax(0), jmax = P.imax(1), kmax = P.imax(2);

  constexpr int dbloc = 100;
  
  for (i=imin; i<imax; i+=dbloc)
    for (j=jmin; j<jmax; j+=dbloc)
      for (k=kmin; k<kmax; k+=dbloc)
#pragma omp task default(shared)
	{
	  int ii = i+dbloc>imax ? imax : i+dbloc;
	  int jj = j+dbloc>jmax ? jmax : j+dbloc;
	  int kk = k+dbloc>kmax ? kmax : k+dbloc;
	  du = iterate_bloc(i, ii, j, jj, k, kk,
		       di, dj, dk,
		       u1, u2, lambda,  mu);
#pragma omp atomic
	  du_sum += du;
        }

#pragma omp taskwait
  return du_sum;
}



