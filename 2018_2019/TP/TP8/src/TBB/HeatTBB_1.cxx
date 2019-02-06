#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

#include "timer.hxx"
#include "Matrix.hxx"
#include "heat.hxx"

#include "tbb/tbb.h"

class cIteration {
public:

  cIteration(Matrix & vv,
	     const Matrix & uu,
	     const Matrix & ff,
	     double ll,
	     double ddtt) : u(uu), v(vv), f(ff), lambda(ll), dt(ddtt) {}
  
  void operator() ( const tbb::blocked_range2d<int, int>& r ) const {
    
    int i,j;
    for (i=r.rows().begin(); i<r.rows().end(); i++)
      for (j=r.cols().begin(); j<r.cols().end(); j++)
    	v(i,j) = u(i,j)
	  - lambda * (4*u(i,j)
		      - u(i+1,j) - u(i-1,j)
		      - u(i,j+1) - u(i,j-1))
	  + f(i,j) * dt;    
  }
  
private:
  Matrix & v;
  const Matrix & u, &f;
  double lambda, dt; 
};

class cDifference {
public:

  cDifference(const Matrix & uu,
	      const Matrix & vv)
    : u(uu), v(vv), diff(0.0) {}
  
  cDifference(cDifference& x, tbb::split )
    : u(x.u), v(x.v), diff(0.0) {}
  
  void operator() ( const tbb::blocked_range2d<int, int>& r ) {
    
    int i,j;
    double local_diff = 0.0;
    for (i=r.rows().begin(); i<r.rows().end(); i++)
      for (j=r.cols().begin(); j<r.cols().end(); j++)
	local_diff += std::abs(v(i,j) - u(i,j));

    diff = local_diff;
  } 

  void join(const cDifference & y ) {diff += y.diff; }

  double result() const { return diff; }
  
private:
  const Matrix & v;
  const Matrix & u;
  double diff; 
};

void Iteration(Matrix &v, const Matrix &u, const Matrix &f,
	       double lambda, double dt)
{
  cIteration It(v, u, f, lambda, dt);
  tbb::blocked_range2d<int, int> Indices(1, u.n()-1, 40, 1, u.m()-1, 40);
  
  tbb:parallel_for(Indices, It);
}

double Difference(const Matrix &v, const Matrix &u)
{  
  cDifference Dif(v, u);
  tbb::blocked_range2d<int, int> Indices(1, u.n()-1, 40, 1, u.m()-1, 40);
  
  tbb:parallel_reduce(Indices, Dif);

  return Dif.result();
}
