#include "Heat.hxx"

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

void Solver::Iteration() {

  cIteration It(m_v, m_u, m_f, m_lambda, m_dt);
  tbb::blocked_range2d<int, int> 
       Indices(1, m_u.n()-1, 40, 1, m_u.m()-1, 40);
  
  It(Indices);
  m_t += m_dt;

}

