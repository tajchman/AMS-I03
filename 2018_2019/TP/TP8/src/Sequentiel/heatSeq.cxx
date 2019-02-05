#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include "Matrix.hxx"

void Iteration(Matrix &v, const Matrix &u, const Matrix &f,
	       double lambda, double dt) {
  
  int i, j, n = u.n(), m = u.m();
  
  for (i=1; i<n-1; i++)
    for (j=1; j<n-1; j++) {
      v(i,j) = u(i,j)
	- lambda * (4*u(i,j)
		    - u(i+1,j) - u(i-1,j)
		    - u(i,j+1) - u(i,j-1)
		    ) + f(i,j) * dt;
    }
}

double Difference(const Matrix &v, const Matrix &u) {
  
  int i, j, n = u.n(), m = u.m();
  double diff = 0.0;
  
  for (i=0; i<n; i++)
    for (j=0; j<n; j++) {
      diff += std::abs(v(i,j) - u(i,j));
    }

  return diff;
}
